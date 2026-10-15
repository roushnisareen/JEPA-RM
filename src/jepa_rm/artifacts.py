"""Where every artifact lives, and syncing it with the Hugging Face Hub.

All stages share one private HF model repo, which is what lets them run on different Modal
accounts. Layout:

    rm_bt/                                         Bradley-Terry reward model (LoRA adapter)
    rm_jepa/                                       JEPA reward model
    policy_{policy}_{run}[_s{seed}]/checkpoint-{step}   policy adapters, run = bt | jepa | dpo
    results/                                       metrics, logs and plots

Seed 0 has no suffix, so a single-seed experiment has the plain names above.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import re
import shutil
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol, TypeVar

import requests
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError

log = logging.getLogger(__name__)
T = TypeVar("T")

RESULTS_DIR = "results"


def seed_suffix(seed: int) -> str:
    return "" if seed == 0 else f"_s{seed}"


def rm_dir(kind: str) -> str:
    return f"rm_{kind}"


def policy_dir(policy: str, run: str, seed: int = 0) -> str:
    return f"policy_{policy}_{run}{seed_suffix(seed)}"


def checkpoint_dir(policy: str, run: str, step: int, seed: int = 0) -> str:
    return f"{policy_dir(policy, run, seed)}/checkpoint-{step}"


def grpo_log_name(policy: str, run: str, seed: int = 0) -> str:
    return f"grpo_{policy}_{run}{seed_suffix(seed)}.json"


def dpo_log_name(policy: str, seed: int = 0) -> str:
    return f"dpo_{policy}{seed_suffix(seed)}.json"


def parse_checkpoints(files: Iterable[str], prefix: str) -> list[tuple[int, str]]:
    """(step, folder) for every adapter saved under `prefix/checkpoint-N`, sorted by step."""
    pattern = re.compile(rf"{re.escape(prefix)}/checkpoint-(\d+)/adapter_config\.json")
    found = []
    for f in files:
        if m := pattern.fullmatch(f):
            found.append((int(m.group(1)), f"{prefix}/checkpoint-{m.group(1)}"))
    return sorted(found)


def is_transient(error: Exception) -> bool:
    """Network blips, timeouts, rate limits and server errors: worth retrying. Anything else
    (missing file, bad token, wrong repo name) won't fix itself, so it isn't."""
    if isinstance(error, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return error.response.status_code == 429 or error.response.status_code >= 500
    return False


def with_retries(fn: Callable[[], T], what: str, attempts: int = 4, wait: float = 10.0) -> T:
    """Call `fn`, retrying transient failures with exponential backoff (10s, 20s, 40s by default)."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts or not is_transient(e):
                raise
            delay = wait * 2 ** (attempt - 1)
            log.warning("%s failed (%s), retry %d/%d in %.0fs", what, e, attempt, attempts - 1, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


class MissingArtifactError(FileNotFoundError):
    """A stage asked for something an earlier stage should have produced."""


def _missing(remote_dir: str, where: str) -> MissingArtifactError:
    if remote_dir.startswith("rm_"):
        hint = "run `modal run modal_app.py::reward_models` first"
    elif remote_dir.startswith("policy_"):
        hint = "run the grpo or dpo job for it first"
    else:
        hint = "check the earlier stages finished"
    return MissingArtifactError(f"{remote_dir}/ not found in {where}: {hint}")


class Store(Protocol):
    """Shared storage for every stage. HubStore on Modal; LocalStore for tests and single-machine runs."""

    def upload_dir(self, local_dir: str | Path, remote_dir: str, allow_patterns: list[str] | None = None) -> None: ...
    def download_dir(self, remote_dir: str) -> Path: ...
    def delete_dir(self, remote_dir: str) -> bool: ...
    def checkpoints(self, policy: str, run: str, seed: int = 0) -> list[tuple[int, str]]: ...
    def load_result(self, name: str) -> dict | None: ...
    def save_result(self, name: str, content: dict | list | bytes) -> None: ...


class LocalStore:
    """The same layout as HubStore, in a folder on disk."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def upload_dir(self, local_dir: str | Path, remote_dir: str, allow_patterns: list[str] | None = None) -> None:
        dest = self.root / remote_dir
        for src in Path(local_dir).rglob("*"):
            rel = src.relative_to(local_dir).as_posix()
            if src.is_file() and (allow_patterns is None or any(fnmatch.fnmatch(rel, p) for p in allow_patterns)):
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / rel)

    def download_dir(self, remote_dir: str) -> Path:
        path = self.root / remote_dir
        if not path.is_dir():
            raise _missing(remote_dir, str(self.root))
        return path

    def delete_dir(self, remote_dir: str) -> bool:
        path = self.root / remote_dir
        if not path.exists():
            return False
        shutil.rmtree(path)
        return True

    def checkpoints(self, policy: str, run: str, seed: int = 0) -> list[tuple[int, str]]:
        files = (p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())
        return parse_checkpoints(files, policy_dir(policy, run, seed))

    def load_result(self, name: str) -> dict | None:
        path = self.root / RESULTS_DIR / name
        return json.loads(path.read_text()) if path.exists() else None

    def save_result(self, name: str, content: dict | list | bytes) -> None:
        path = self.root / RESULTS_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else json.dumps(content, indent=2).encode())


class HubStore:
    """A private HF repo used as shared storage for every stage. Every call retries transient
    network errors, since one failed request shouldn't cost a multi-hour run."""

    def __init__(self, repo: str):
        self.repo = repo
        self.api = HfApi()
        with_retries(lambda: self.api.create_repo(repo, private=True, exist_ok=True), f"create {repo}")

    def upload_dir(self, local_dir: str | Path, remote_dir: str, allow_patterns: list[str] | None = None) -> None:
        with_retries(lambda: self.api.upload_folder(
            folder_path=str(local_dir), repo_id=self.repo, path_in_repo=remote_dir, allow_patterns=allow_patterns
        ), f"upload {remote_dir}")

    def download_dir(self, remote_dir: str) -> Path:
        root = with_retries(lambda: snapshot_download(self.repo, allow_patterns=[f"{remote_dir}/*"]),
                            f"download {remote_dir}")
        path = Path(root) / remote_dir
        if not path.is_dir():
            raise _missing(remote_dir, self.repo)
        return path

    def _files(self) -> list[str]:
        return with_retries(lambda: self.api.list_repo_files(self.repo), f"list {self.repo}")

    def delete_dir(self, remote_dir: str) -> bool:
        if not any(f.startswith(f"{remote_dir}/") for f in self._files()):
            return False
        with_retries(lambda: self.api.delete_folder(
            remote_dir, repo_id=self.repo, commit_message=f"clear {remote_dir} before a new run"
        ), f"delete {remote_dir}")
        return True

    def checkpoints(self, policy: str, run: str, seed: int = 0) -> list[tuple[int, str]]:
        return parse_checkpoints(self._files(), policy_dir(policy, run, seed))

    def load_result(self, name: str) -> dict | None:
        """A JSON result saved by an earlier job, or None if that job hasn't run yet."""
        try:
            path = with_retries(lambda: hf_hub_download(self.repo, f"{RESULTS_DIR}/{name}"), f"download {name}")
        except EntryNotFoundError:
            return None
        return json.loads(Path(path).read_text())

    def save_result(self, name: str, content: dict | list | bytes) -> None:
        data = content if isinstance(content, bytes) else json.dumps(content, indent=2).encode()
        with_retries(lambda: self.api.upload_file(
            path_or_fileobj=data, path_in_repo=f"{RESULTS_DIR}/{name}", repo_id=self.repo
        ), f"upload {name}")
