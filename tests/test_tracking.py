import pytest
import wandb

from jepa_rm import tracking


@pytest.fixture
def offline_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")


def test_logging_is_a_no_op_without_a_run():
    assert not tracking.active()
    tracking.define_axis("bt", "bt/step")
    tracking.log({"bt/loss": 1.0})
    tracking.summary({"x": 1})
    tracking.log_table("t", [{"a": 1}])
    tracking.log_image("img", b"not a png")  # never decoded without a run


def test_run_logs_and_finishes(offline_wandb):
    with tracking.run("test", job_type="test", config={"lr": 1e-4}):
        assert tracking.active()
        assert wandb.run.config["lr"] == 1e-4
        tracking.define_axis("bt", "bt/step")
        tracking.log({"bt/step": 1, "bt/loss": 0.5})
        tracking.log_table("t", [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        tracking.summary({"final": 0.9})
    assert not tracking.active()


def test_run_finishes_even_when_the_job_fails(offline_wandb):
    with pytest.raises(RuntimeError), tracking.run("test", job_type="test", config={}):
        raise RuntimeError("boom")
    assert not tracking.active()
