"""`load_encoder` is the single seam both the corpus and the query side load
through, so what it passes is what makes the two comparable."""
import pytest

from process.embedder import MODELS, load_encoder


class FakeModule:
    def __init__(self):
        self.auto_model = None


class FakeEncoder:
    """Stands in for SentenceTransformer, recording how it was constructed."""

    def __init__(self, hf_name, device, trust_remote_code, config_kwargs):
        self.hf_name = hf_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.config_kwargs = config_kwargs

    def __getitem__(self, index):
        return FakeModule()


@pytest.fixture
def fake_encoder(monkeypatch):
    built = {}

    def factory(hf_name, device, trust_remote_code, config_kwargs):
        built["encoder"] = FakeEncoder(
            hf_name, device, trust_remote_code, config_kwargs
        )
        return built["encoder"]

    monkeypatch.setattr("process.embedder.SentenceTransformer", factory)
    return built


class TestLoadEncoder:
    def test_plain_model_loads_without_remote_code(self, fake_encoder):
        encoder = load_encoder("bge-small", "cpu")

        assert encoder.hf_name == "BAAI/bge-small-en-v1.5"
        assert encoder.device == "cpu"
        assert encoder.trust_remote_code is False
        assert encoder.config_kwargs == {}

    def test_arctic_disables_the_broken_xformers_path(self, fake_encoder, monkeypatch):
        repaired = []
        monkeypatch.setattr(
            "process.embedder._repair_gte_buffers", lambda m: repaired.append(m)
        )

        encoder = load_encoder("arctic-m-v2", "cuda")

        assert encoder.trust_remote_code is True
        assert encoder.config_kwargs == {
            "use_memory_efficient_attention": False,
            "unpad_inputs": False,
        }
        # Without the repair the model returns NaN or asserts on the device,
        # depending on batch shape — so it has to run on every load, not once
        # by hand in a notebook.
        assert repaired == [encoder]

    def test_models_without_the_fault_are_not_repaired(self, fake_encoder, monkeypatch):
        repaired = []
        monkeypatch.setattr(
            "process.embedder._repair_gte_buffers", lambda m: repaired.append(m)
        )

        load_encoder("bge-large", "cpu")

        assert repaired == []


class TestModelRegistry:
    @pytest.mark.parametrize("key", sorted(MODELS))
    def test_every_model_declares_what_the_index_needs(self, key):
        cfg = MODELS[key]

        assert cfg["embed_size"] > 0
        # A wrong embed_size creates a Pinecone index that rejects every
        # upsert, and the index name is what keeps two models' vectors apart.
        assert cfg["index_name"].startswith("papers-please-")
        assert cfg["query_prompt"]
