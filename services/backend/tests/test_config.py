"""Unit tests for per-stage config parsing (config.yaml -> Config)."""
import textwrap

from config import Config


def test_stages_config_defaults():
    cfg = Config()
    assert cfg.stages.download.interval_s == 300
    assert cfg.stages.download.limit == 20
    assert cfg.stages.download.workers == 4
    assert cfg.stages.chunk.interval_s == 300
    assert cfg.stages.chunk.limit == 10
    assert cfg.stages.embed.interval_s == 300
    assert cfg.stages.embed.limit == 500


def test_stages_config_parses_independent_intervals(tmp_path):
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            """
            stages:
              download:
                interval_s: 5
                limit: 20
                workers: 4
              chunk:
                interval_s: 15
                limit: 10
              embed:
                interval_s: 60
                limit: 500
            """
        )
    )
    data = yaml.safe_load(path.read_text())
    cfg = Config.model_validate(data)

    assert cfg.stages.download.interval_s == 5
    assert cfg.stages.chunk.interval_s == 15
    assert cfg.stages.embed.interval_s == 60
