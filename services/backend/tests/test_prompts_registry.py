"""Unit tests for prompts/registry.py and the config that selects versions.

No LLM involved - this is file lookup plus the guarantee that the v1 files
are what the code used to hardcode, which is what makes the existing
eval/reports/ baselines still comparable.
"""
import pytest

from config import Config
from prompts.registry import PROMPTS_DIR, available_versions, load_prompt


def test_loads_orchestrator_v1():
    prompt = load_prompt("orchestrator", "v1")
    assert prompt.startswith("You're the assistant for a research paper library.")
    assert "cite the" in prompt


def test_loads_fixed_rag_v1():
    prompt = load_prompt("fixed_rag", "v1")
    assert "only the context passages" in prompt
    assert "don't guess or use" in prompt


def test_strips_trailing_whitespace():
    """The file ends with a newline; the prompt sent to the LLM should not -
    a trailing blank line is a (small) unintended part of the prompt."""
    raw = (PROMPTS_DIR / "fixed_rag" / "v1.md").read_text()
    assert raw.endswith("\n")
    assert not load_prompt("fixed_rag", "v1").endswith("\n")


def test_missing_version_raises_with_available_listed():
    with pytest.raises(FileNotFoundError) as exc:
        load_prompt("orchestrator", "v99")
    assert "v1" in str(exc.value)


def test_missing_prompt_name_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("no_such_prompt", "v1")


def test_available_versions_sorts_numerically(tmp_path, monkeypatch):
    """v10 must sort after v2 - a plain string sort puts it first, which
    would make 'the latest version' read off this list wrong."""
    d = tmp_path / "fake_prompt"
    d.mkdir()
    for v in ("v1", "v2", "v10"):
        (d / f"{v}.md").write_text("x")
    monkeypatch.setattr("prompts.registry.PROMPTS_DIR", tmp_path)
    assert available_versions("fake_prompt") == ["v1", "v2", "v10"]


def test_available_versions_empty_for_unknown_name():
    assert available_versions("no_such_prompt") == []


def test_config_defaults_to_v1():
    cfg = Config()
    assert cfg.prompts.orchestrator == "v1"
    assert cfg.prompts.fixed_rag == "v1"


def test_config_overrides_prompt_versions():
    cfg = Config.model_validate({"prompts": {"orchestrator": "v3"}})
    assert cfg.prompts.orchestrator == "v3"
    assert cfg.prompts.fixed_rag == "v1"  # untouched key keeps its default


def test_every_configured_version_exists_on_disk():
    """Guards the failure mode where config names a version nobody created -
    which would only surface at agent-build time, in production."""
    for name, version in Config().prompts.model_dump().items():
        assert load_prompt(name, version)
