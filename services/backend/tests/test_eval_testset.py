"""Test-set generation, and the two ragas behaviours that fail silently.

Both would produce a plausible-looking test set and a clean run:

  1. `default_transforms` inserts a HeadlineSplitter when >=25% of inputs
     exceed 500 tokens, re-splitting chunks that are already the retrieval
     unit and breaking the chunk-id mapping the free metrics depend on.
  2. `default_query_distribution` drops any synthesizer whose clusters are
     empty, so a 50/25/25 split can quietly become 100/0/0.

No API calls anywhere here - the LLM and embeddings are stubs, and the
transforms are inspected rather than applied.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from eval.testset import (
    DISTRIBUTION,
    MIN_CHUNK_TOKENS,
    assert_clusters,
    map_chunk_ids,
    query_distribution,
    seed_pool,
    to_rows,
    extraction_transforms,
    relationship_transforms,
)


class TestTransforms:
    def test_never_includes_a_splitter(self):
        """A splitter would re-cut our chunks, and a node whose text no
        longer matches chunks.chunk_text can never be mapped back to an id."""
        from ragas.testset.transforms import HeadlineSplitter, Parallel

        flat = []
        for t in extraction_transforms(MagicMock(), MagicMock()) + relationship_transforms():
            flat.extend(t.transformations if isinstance(t, Parallel) else [t])

        assert not any(isinstance(t, HeadlineSplitter) for t in flat)

    def test_builds_the_relationship_each_synthesizer_needs(self):
        """summary_similarity feeds multi-hop abstract, entities_overlap
        feeds multi-hop specific. Missing either drops a synthesizer."""
        from ragas.testset.transforms import (
            CosineSimilarityBuilder,
            OverlapScoreBuilder,
            Parallel,
        )

        flat = []
        for t in extraction_transforms(MagicMock(), MagicMock()) + relationship_transforms():
            flat.extend(t.transformations if isinstance(t, Parallel) else [t])

        cosine = next(t for t in flat if isinstance(t, CosineSimilarityBuilder))
        assert cosine.new_property_name == "summary_similarity"
        assert any(isinstance(t, OverlapScoreBuilder) for t in flat)

    def test_similarity_threshold_is_tuned_to_this_corpus(self):
        """Neither ragas default fits. 0.9 pairs almost nothing; 0.5 admitted
        94.7% of all possible pairs on 100 ML papers, making a complete graph
        whose depth-3 cluster search never returns - it hung a run for 13
        minutes at 0% CPU."""
        from eval.testset import COSINE_THRESHOLD
        from ragas.testset.transforms import CosineSimilarityBuilder, Parallel

        flat = []
        for t in relationship_transforms():
            flat.extend(t.transformations if isinstance(t, Parallel) else [t])
        builder = next(t for t in flat if isinstance(t, CosineSimilarityBuilder))

        assert builder.threshold == COSINE_THRESHOLD == 0.80


class TestAssertClusters:
    def _kg(self, counts):
        return counts

    def test_passes_when_every_synthesizer_has_clusters(self):
        with patch("eval.testset.cluster_counts", return_value={"a": 3, "b": 1}):
            assert assert_clusters(MagicMock()) == {"a": 3, "b": 1}

    def test_raises_naming_the_empty_synthesizer(self):
        """The whole point: ragas would drop it and still return 130
        plausible questions."""
        with patch(
            "eval.testset.cluster_counts",
            return_value={"single_hop_specific": 400, "multi_hop_abstract": 0},
        ):
            with pytest.raises(RuntimeError, match="multi_hop_abstract"):
                assert_clusters(MagicMock())

    def test_reports_every_empty_synthesizer_not_just_the_first(self):
        with patch(
            "eval.testset.cluster_counts",
            return_value={"a": 1, "multi_hop_abstract": 0, "multi_hop_specific": 0},
        ):
            with pytest.raises(RuntimeError) as e:
                assert_clusters(MagicMock())

        assert "multi_hop_abstract" in str(e.value) and "multi_hop_specific" in str(e.value)


class TestQueryDistribution:
    def test_weights_are_the_requested_split_not_ragas_thirds(self):
        dist = query_distribution(MagicMock())

        assert [w for _, w in dist] == [0.5, 0.25, 0.25]

    def test_covers_all_three_synthesizers(self):
        assert set(DISTRIBUTION) == {
            "single_hop_specific",
            "multi_hop_specific",
            "multi_hop_abstract",
        }

    def test_weights_sum_to_one(self):
        assert sum(DISTRIBUTION.values()) == 1.0


class TestMapChunkIds:
    def test_resolves_by_exact_text(self):
        """Exact equality, not similarity - no splitter ran, so the node text
        is byte-identical to chunks.chunk_text."""
        assert map_chunk_ids(["abc", "def"], {"abc": 1, "def": 2}) == [1, 2]

    def test_drops_a_context_that_does_not_resolve(self):
        assert map_chunk_ids(["abc", "mangled"], {"abc": 1}) == [1]

    def test_preserves_order(self):
        assert map_chunk_ids(["b", "a"], {"a": 1, "b": 2}) == [2, 1]


class TestToRows:
    def _testset(self, contexts, name="single_hop_specific_query_synthesizer"):
        sample = SimpleNamespace(
            eval_sample=SimpleNamespace(
                user_input="q?", reference="a", reference_contexts=contexts
            ),
            synthesizer_name=name,
        )
        return SimpleNamespace(samples=[sample])

    def _pool(self):
        return [
            {"chunk_id": 7, "doc_id": 3, "topic": "retrieval", "text": "alpha"},
            {"chunk_id": 8, "doc_id": 4, "topic": "agents", "text": "beta"},
        ]

    def test_carries_chunk_doc_and_topic_ids(self):
        rows, _ = to_rows(self._testset(["alpha", "beta"]), self._pool())

        assert rows[0]["reference_chunk_ids"] == [7, 8]
        assert rows[0]["reference_doc_ids"] == [3, 4]

    def test_topics_are_deduped_and_sorted(self):
        rows, _ = to_rows(self._testset(["alpha", "beta"]), self._pool())

        assert rows[0]["topics"] == ["agents", "retrieval"]

    def test_counts_contexts_that_failed_to_map(self):
        """A silently empty reference_chunk_ids would read as 'retrieval
        found nothing' on every run, forever."""
        _, unmapped = to_rows(self._testset(["alpha", "not in pool"]), self._pool())

        assert unmapped == 1

    def test_ids_are_stable_and_zero_padded(self):
        rows, _ = to_rows(self._testset(["alpha"]), self._pool())

        assert rows[0]["id"] == "q0000"


class TestSeedPool:
    def _rows(self, n_per_topic, text="word " * 200):
        rows = []
        cid = 0
        for topic in ("retrieval", "agents"):
            for _ in range(n_per_topic):
                cid += 1
                rows.append(
                    SimpleNamespace(id=cid, chunk_text=text, doc_id=cid, topic=topic)
                )
        return rows

    def _run(self, rows, per_topic):
        session = MagicMock()
        session.execute.return_value.all.return_value = rows
        with (
            patch("eval.testset.Session") as sess,
            patch("db.connection.PostgresInterface.connect"),
        ):
            sess.return_value.__enter__.return_value = session
            return seed_pool(per_topic=per_topic)

    def test_samples_per_topic_not_across_the_whole_corpus(self):
        """Chunk counts differ several-fold between papers; a global sample
        would let one long survey crowd out an entire topic."""
        pool = self._run(self._rows(50), per_topic=10)

        by_topic = {}
        for c in pool:
            by_topic[c["topic"]] = by_topic.get(c["topic"], 0) + 1
        assert by_topic == {"retrieval": 10, "agents": 10}

    def test_takes_everything_when_a_topic_is_short(self):
        assert len(self._run(self._rows(3), per_topic=10)) == 6

    def test_drops_chunks_below_the_summariser_floor(self):
        """A chunk under 100 tokens gets no summary, so it can never join a
        summary_similarity cluster - it is dead weight in the graph."""
        short = self._rows(5, text="tiny")

        assert self._run(short, per_topic=10) == []

    def test_is_deterministic_across_runs(self):
        """Two runs of the same experiment must draw from the same pool, or
        they are not the same experiment."""
        rows = self._rows(50)

        assert [c["chunk_id"] for c in self._run(rows, 10)] == [
            c["chunk_id"] for c in self._run(rows, 10)
        ]

    def test_floor_matches_the_extractor_gate(self):
        assert MIN_CHUNK_TOKENS == 100


class TestRunConfig:
    def test_drives_the_provider_less_hard_than_ragas_defaults(self):
        """ragas defaults to 16 workers and 10 retries. Against OpenRouter
        that is the expensive failure mode: concurrency trips the rate limit
        and ten retries keep paying rather than failing visibly."""
        from eval.testset import run_config

        rc = run_config()

        assert rc.max_workers < 16 and rc.max_retries < 10

    def test_reads_the_limits_from_config(self):
        import config as config_module
        from config import Config, RagasConfig
        from eval.testset import run_config

        original = config_module._config
        try:
            config_module._config = Config(ragas=RagasConfig(max_workers=2, max_retries=1))
            rc = run_config()
        finally:
            config_module._config = original

        assert (rc.max_workers, rc.max_retries) == (2, 1)


class TestCostAccounting:
    def test_handler_is_attached_to_the_langchain_model(self):
        """apply_transforms takes a `callbacks` argument and never uses it,
        so a handler passed to ragas collects nothing from the graph build -
        which is the bulk of the spend. It has to ride on the model."""
        from unittest.mock import MagicMock, patch

        from eval.testset import generator_llm

        handler = MagicMock()
        chat = MagicMock()
        with (
            patch("orchestrator.llm.openrouter_chat", return_value=chat),
            patch("ragas.llms.LangchainLLMWrapper", side_effect=lambda c: MagicMock()),
        ):
            generator_llm(handler)

        assert chat.callbacks == [handler]

    def test_reports_nothing_rather_than_zero_when_no_calls_were_made(self, capsys):
        """A silent $0.0000 is indistinguishable from working accounting on
        a run that spent money."""
        from unittest.mock import MagicMock

        from eval.testset import report_spend

        handler = MagicMock(usage_data=[])
        report_spend(handler, "z-ai/glm-5.3-flash", "graph")

        assert "no usage recorded" in capsys.readouterr().out

    def test_reports_dollars_for_a_priced_model(self, capsys):
        from unittest.mock import MagicMock

        from eval.testset import report_spend
        from ragas.cost import TokenUsage

        usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, model="x")
        handler = MagicMock(usage_data=[usage], total_tokens=lambda: usage)
        report_spend(handler, "z-ai/glm-5.3-flash", "graph")

        assert "$0.0750" in capsys.readouterr().out

    def test_reports_tokens_without_dollars_for_an_unpriced_model(self, capsys):
        from unittest.mock import MagicMock

        from eval.testset import report_spend
        from ragas.cost import TokenUsage

        usage = TokenUsage(input_tokens=10, output_tokens=5, model="x")
        handler = MagicMock(usage_data=[usage], total_tokens=lambda: usage)
        report_spend(handler, "who/knows", "graph")

        out = capsys.readouterr().out
        assert "no price on record" in out and "$" not in out


class TestPersonas:
    def test_missing_file_is_not_an_error(self, tmp_path, monkeypatch):
        """The graph stage writes them; generate must still run before it."""
        import eval.testset as ts

        monkeypatch.setattr(ts, "PERSONAS_PATH", tmp_path / "absent.json")

        assert ts.load_personas() is None

    def test_round_trips_through_disk(self, tmp_path, monkeypatch):
        """Persisted so a regenerated test set is the same test set - new
        personas mean differently-voiced questions."""
        import json

        import eval.testset as ts

        path = tmp_path / "personas.json"
        path.write_text(json.dumps([{"name": "n", "role_description": "r"}]))
        monkeypatch.setattr(ts, "PERSONAS_PATH", path)

        assert ts.load_personas()[0].name == "n"


class TestPermissiveTokenizer:
    """A corpus of LLM papers quotes `<|endoftext|>` in running prose, and
    tiktoken's default raises on it. `LLMBasedExtractor.split_text_by_token_limit`
    calls `tokenizer.encode` unconditionally on every node, so the graph build
    would die partway through - after paying for every node before it."""

    def test_counts_text_containing_a_special_token_literal(self):
        from eval.testset import count_tokens

        assert count_tokens("hello <|endoftext|> world") > 0

    def test_ragas_own_counter_would_have_raised(self):
        """Pins why count_tokens exists rather than reusing ragas'."""
        import pytest
        from ragas.utils import num_tokens_from_string

        with pytest.raises(ValueError, match="disallowed special token"):
            num_tokens_from_string("hello <|endoftext|> world")

    def test_encode_decode_round_trips_the_literal(self):
        """The marker must survive: stripping it would make a node's
        page_content differ from chunks.chunk_text and break the chunk-id
        mapping the free metrics rest on."""
        from eval.testset import permissive_tokenizer

        tok = permissive_tokenizer()

        assert tok.decode(tok.encode("a <|endoftext|> b")) == "a <|endoftext|> b"

    def test_llm_extractors_get_the_permissive_tokenizer(self):
        from unittest.mock import MagicMock

        from eval.testset import PermissiveTokenizer
        from ragas.testset.transforms import Parallel
        from ragas.testset.transforms.extractors.llm_based import LLMBasedExtractor

        flat = []
        for t in extraction_transforms(MagicMock(), MagicMock()):
            flat.extend(t.transformations if isinstance(t, Parallel) else [t])
        llm_based = [t for t in flat if isinstance(t, LLMBasedExtractor)]

        assert llm_based and all(
            isinstance(t.tokenizer, PermissiveTokenizer) for t in llm_based
        )


class TestPruneIncomplete:
    """One failed extraction must not cost a whole relationship type.
    OverlapScoreBuilder raises on the first pair involving a node with no
    entities and aborts entirely - which is how a run produced 75,223 cosine
    relationships and zero entities_overlap ones, silently removing every
    multi-hop-specific question from a test set that still looked complete."""

    def _kg(self, *nodes):
        from ragas.testset.graph import KnowledgeGraph

        kg = KnowledgeGraph()
        for n in nodes:
            kg.add(n)
        return kg

    def _node(self, **props):
        from ragas.testset.graph import Node, NodeType

        return Node(type=NodeType.DOCUMENT, properties={"page_content": "t", **props})

    def test_drops_a_node_with_no_entities(self):
        from eval.testset import prune_incomplete

        good = self._node(entities=["a"], summary_embedding=[0.1])
        bad = self._node(summary_embedding=[0.1])
        kg = self._kg(good, bad)

        prune_incomplete(kg)

        assert [n.id for n in kg.nodes] == [good.id]

    def test_drops_a_node_with_no_summary_embedding(self):
        from eval.testset import prune_incomplete

        good = self._node(entities=["a"], summary_embedding=[0.1])
        kg = self._kg(good, self._node(entities=["b"]))

        prune_incomplete(kg)

        assert [n.id for n in kg.nodes] == [good.id]

    def test_keeps_a_complete_graph_intact(self):
        from eval.testset import prune_incomplete

        kg = self._kg(self._node(entities=["a"], summary_embedding=[0.1]))

        assert prune_incomplete(kg) == [] and len(kg.nodes) == 1


class TestGeneratorTokenBudget:
    """glm-5.3-flash is a reasoning model: Phoenix measured 342 reasoning
    tokens out of 350 completion tokens on a typical call. At 2048 the
    theme/persona matching prompt ran out mid-thought and raised
    LLMDidNotFinishException, killing generation outright."""

    def test_budget_leaves_room_for_reasoning_tokens(self):
        from config import RagasConfig

        assert RagasConfig().generator_max_tokens >= 8192

    def test_generator_uses_the_configured_budget(self):
        from unittest.mock import MagicMock, patch

        import config as config_module
        from config import Config, RagasConfig
        from eval.testset import generator_llm

        original = config_module._config
        try:
            config_module._config = Config(ragas=RagasConfig(generator_max_tokens=4321))
            with (
                patch("orchestrator.llm.openrouter_chat") as chat,
                patch("ragas.llms.LangchainLLMWrapper", side_effect=lambda c: MagicMock()),
            ):
                generator_llm()
        finally:
            config_module._config = original

        assert chat.call_args.args[1] == 4321


class TestHopMarkers:
    """Multi-hop synthesizers prefix each context with the hop it came from.
    An exact lookup misses every one: on a 7-question trial this emptied
    reference_chunk_ids for all four multi-hop rows - 8 of 11 contexts
    unresolved - which downstream reads as "retrieval found nothing" on every
    run forever."""

    def test_strips_the_one_hop_marker(self):
        from eval.testset import map_chunk_ids

        assert map_chunk_ids(["<1-hop>\n\nalpha"], {"alpha": 7}) == [7]

    def test_strips_higher_hop_markers(self):
        from eval.testset import map_chunk_ids

        assert map_chunk_ids(["<2-hop>\n\nbeta"], {"beta": 8}) == [8]

    def test_leaves_single_hop_text_untouched(self):
        from eval.testset import map_chunk_ids

        assert map_chunk_ids(["alpha"], {"alpha": 7}) == [7]

    def test_does_not_strip_a_marker_mid_text(self):
        """Only a leading marker is ragas'. One inside the passage is the
        paper's own prose and must not be edited away."""
        from eval.testset import map_chunk_ids

        assert map_chunk_ids(["see <1-hop>\n\nhere"], {"see <1-hop>\n\nhere": 9}) == [9]

    def test_still_drops_a_genuinely_unresolvable_context(self):
        from eval.testset import map_chunk_ids

        assert map_chunk_ids(["<1-hop>\n\nmangled"], {"alpha": 7}) == []
