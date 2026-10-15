"""Train every reward model kind in one go, the way the `reward_models` job does.

The embedding-space kinds share one encoder and one embedding pass, which is what makes them a
fair comparison (and makes the extra controls cost minutes, not hours).
"""
from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from jepa_rm.artifacts import rm_dir
from jepa_rm.config import BTConfig, JepaConfig
from jepa_rm.data import PreferencePair
from jepa_rm.rewards import BT_KINDS, JEPA_VARIANTS, RewardModel, load_reward_model
from jepa_rm.rewards.bradley_terry import make_length_penalized, train_bradley_terry
from jepa_rm.rewards.embeddings import Encoder, embed_pairs
from jepa_rm.rewards.frozen_bt import train_frozen_bt
from jepa_rm.rewards.jepa import train_jepa

LENGTH_FIT_PAIRS = 1000  # training pairs used to fit the length penalty's slope


def train_reward_models(
    pairs: Sequence[PreferencePair], out_root: str | Path, bt_cfg: BTConfig, jepa_cfg: JepaConfig,
    device: str | None = None, bt_from: str | Path | None = None,
) -> dict[str, RewardModel]:
    """Each kind is saved to out_root/rm_{kind}; returns the trained models by kind.

    `bt_from` is a folder holding rm_bt/ and rm_bt_lenpen/ from an earlier run. They're copied rather
    than retrained: BT doesn't depend on the embedding settings, so a pooling ablation then compares
    the embedding-space kinds against the very same BT.
    """
    out_root = Path(out_root)
    models: dict[str, RewardModel] = {}

    if bt_from is not None:
        for kind in BT_KINDS:
            shutil.copytree(Path(bt_from) / rm_dir(kind), out_root / rm_dir(kind), dirs_exist_ok=True)
            models[kind] = load_reward_model(kind, out_root / rm_dir(kind))
    else:
        bt = train_bradley_terry(pairs, out_root / rm_dir("bt"), bt_cfg, device)
        models["bt"] = bt
        models["bt_lenpen"] = make_length_penalized(
            bt, out_root / rm_dir("bt"), out_root / rm_dir("bt_lenpen"), pairs[:LENGTH_FIT_PAIRS]
        )

    encoder = Encoder.from_config(jepa_cfg, device)
    embeddings = embed_pairs(encoder, pairs, jepa_cfg.holdout)
    models["frozen_bt"] = train_frozen_bt(pairs, out_root / rm_dir("frozen_bt"), jepa_cfg,
                                          encoder=encoder, embeddings=embeddings)
    for kind, overrides in JEPA_VARIANTS.items():
        models[kind] = train_jepa(pairs, out_root / rm_dir(kind), replace(jepa_cfg, **overrides),
                                  encoder=encoder, embeddings=embeddings, tag=kind)
    return models
