import pytest
import requests

from jepa_rm.artifacts import (
    LocalStore,
    MissingArtifactError,
    checkpoint_dir,
    dpo_log_name,
    grpo_log_name,
    parse_checkpoints,
    policy_dir,
    with_retries,
)


def test_checkpoints_are_found_and_sorted_by_step():
    files = [
        "policy_qwen_bt/checkpoint-100/adapter_config.json",
        "policy_qwen_bt/checkpoint-50/adapter_config.json",
        "policy_qwen_bt/checkpoint-50/adapter_model.safetensors",
        "policy_qwen_jepa/checkpoint-25/adapter_config.json",
        "rm_bt/adapter_config.json",
        "results/eval_qwen.json",
    ]
    assert parse_checkpoints(files, "policy_qwen_bt") == [
        (50, "policy_qwen_bt/checkpoint-50"),
        (100, "policy_qwen_bt/checkpoint-100"),
    ]


def test_prefix_must_match_exactly():
    files = ["policy_qwen_bt_extra/checkpoint-1/adapter_config.json"]
    assert parse_checkpoints(files, "policy_qwen_bt") == []


def test_checkpoint_dir_round_trips_through_the_parser():
    folder = checkpoint_dir("phi", "jepa", 200)
    assert parse_checkpoints([f"{folder}/adapter_config.json"], policy_dir("phi", "jepa")) == [(200, folder)]


def test_seed_zero_keeps_the_plain_names():
    assert policy_dir("qwen", "bt") == "policy_qwen_bt"
    assert policy_dir("qwen", "bt", seed=2) == "policy_qwen_bt_s2"
    assert grpo_log_name("qwen", "jepa", seed=1) == "grpo_qwen_jepa_s1.json"
    assert dpo_log_name("qwen") == "dpo_qwen.json"


def test_seeds_never_see_each_others_checkpoints():
    files = [f"{checkpoint_dir('qwen', 'bt', 50, seed)}/adapter_config.json" for seed in (0, 1)]
    assert parse_checkpoints(files, policy_dir("qwen", "bt")) == [(50, "policy_qwen_bt/checkpoint-50")]
    assert parse_checkpoints(files, policy_dir("qwen", "bt", 1)) == [(50, "policy_qwen_bt_s1/checkpoint-50")]


def test_missing_artifacts_say_which_job_to_run(tmp_path):
    store = LocalStore(tmp_path)
    with pytest.raises(MissingArtifactError, match="reward_models"):
        store.download_dir("rm_bt")
    with pytest.raises(MissingArtifactError, match="grpo or dpo"):
        store.download_dir("policy_qwen_bt/checkpoint-50")


def test_delete_dir_clears_a_run_and_reports_whether_anything_was_there(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "adapter_config.json").write_text("{}")
    store = LocalStore(tmp_path / "store")
    store.upload_dir(src, checkpoint_dir("qwen", "bt", 8))
    assert store.checkpoints("qwen", "bt") == [(8, "policy_qwen_bt/checkpoint-8")]
    assert store.delete_dir(policy_dir("qwen", "bt")) is True
    assert store.checkpoints("qwen", "bt") == []
    assert store.delete_dir(policy_dir("qwen", "bt")) is False


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(response=response)


class Flaky:
    """Fails with `error` the first `failures` times it's called, then returns "ok"."""

    def __init__(self, failures, error):
        self.failures, self.error, self.calls = failures, error, 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok"


@pytest.mark.parametrize("error", [requests.ConnectionError(), requests.Timeout(), http_error(503), http_error(429)])
def test_transient_errors_are_retried(error):
    flaky = Flaky(failures=2, error=error)
    assert with_retries(flaky, "upload", wait=0) == "ok"
    assert flaky.calls == 3


@pytest.mark.parametrize("error", [http_error(401), http_error(404), ValueError("bug")])
def test_permanent_errors_fail_straight_away(error):
    flaky = Flaky(failures=1, error=error)
    with pytest.raises(type(error)):
        with_retries(flaky, "upload", wait=0)
    assert flaky.calls == 1


def test_retries_give_up_eventually():
    flaky = Flaky(failures=10, error=requests.ConnectionError())
    with pytest.raises(requests.ConnectionError):
        with_retries(flaky, "upload", attempts=3, wait=0)
    assert flaky.calls == 3
