from __future__ import annotations

import argparse
import json

import pytest

from breeze_train import tracking


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("WANDB_API_KEY", "TRACKING_TEST_A", "TRACKING_TEST_B", "TRACKING_TEST_C"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_load_dotenv_parses_and_never_overrides(tmp_path, clean_env):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "export TRACKING_TEST_A=\"two words\"\n"
        "TRACKING_TEST_B='quoted'\n"
        "TRACKING_TEST_C=from-file\n"
        "\n"
        "NOT_AN_ASSIGNMENT\n"
    )
    clean_env.setenv("TRACKING_TEST_C", "from-env")

    loaded = tracking.load_dotenv(env)

    assert loaded == ["TRACKING_TEST_A", "TRACKING_TEST_B"]
    assert tracking.os.environ["TRACKING_TEST_A"] == "two words"
    assert tracking.os.environ["TRACKING_TEST_B"] == "quoted"
    assert tracking.os.environ["TRACKING_TEST_C"] == "from-env"


def test_load_dotenv_missing_file_is_noop(tmp_path):
    assert tracking.load_dotenv(tmp_path / "absent.env") == []


def test_has_credentials_reads_env_and_netrc(tmp_path, clean_env):
    clean_env.setenv("HOME", str(tmp_path))
    assert not tracking.has_credentials()
    netrc = tmp_path / ".netrc"
    netrc.write_text("machine api.wandb.ai\n  login user\n  password secret\n")
    netrc.chmod(0o600)
    assert tracking.has_credentials()
    netrc.unlink()
    clean_env.setenv("WANDB_API_KEY", "x")
    assert tracking.has_credentials()


def test_init_run_is_off_without_project():
    parser = argparse.ArgumentParser()
    tracking.add_wandb_args(parser, attach=True)
    assert tracking.init_run(parser.parse_args([]), job_type="train") is None


def test_init_run_offline_records_and_resumes_run_id(tmp_path, clean_env):
    pytest.importorskip("wandb")
    clean_env.setenv("WANDB_MODE", "offline")
    clean_env.setenv("WANDB_SILENT", "true")
    parser = argparse.ArgumentParser()
    tracking.add_wandb_args(parser)
    args = parser.parse_args(["--wandb-project", "unit-test", "--wandb-tags", "a, b"])

    run = tracking.init_run(args, job_type="train", run_dir=tmp_path, config={"x": 1})
    first_id = run.id
    assert run.config["x"] == 1
    assert "git" in run.config and "host" in run.config
    assert sorted(run.tags) == ["a", "b"]
    run.finish()

    saved = json.loads((tmp_path / tracking.RUN_FILE).read_text())
    assert saved["id"] == first_id
    resumed = tracking.init_run(args, job_type="train", run_dir=tmp_path, resume_from=tmp_path)
    assert resumed.id == first_id
    resumed.finish()


def test_attaching_nests_config_under_job_type(tmp_path, clean_env):
    pytest.importorskip("wandb")
    clean_env.setenv("WANDB_MODE", "offline")
    clean_env.setenv("WANDB_SILENT", "true")
    clean_env.chdir(tmp_path)
    parser = argparse.ArgumentParser()
    tracking.add_wandb_args(parser, attach=True)
    args = parser.parse_args(["--wandb-project", "unit-test", "--wandb-run-id", "abc123"])
    run = tracking.init_run(args, job_type="eval", config={"name": "tuned"})
    assert "git" not in run.config
    assert run.config["eval_context"]["name"] == "tuned"
    run.finish()
