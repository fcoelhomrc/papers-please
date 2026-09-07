"""Unit tests for eval/figures.py.

Split in two: the loading and vocabulary layer is pure and runs anywhere, and
the rendering layer is skipped where matplotlib is not installed (it is a
`--with matplotlib` extra, not a project dependency).

The cases here are the defects the first cut of these figures actually
shipped with - a hardcoded arm list that silently dropped HyDE, subtitles
clipped by the fixed figure width, and results ordered by mtime so that
copying them between checkouts reordered the runs.
"""
import json

import pytest

from eval.figures import (
    FORMATS,
    LABEL,
    MODES,
    THEMES,
    Frame,
    label,
    load_ablation,
    provenance,
)


def summary(**over):
    base = {m: 0.5 for m in ("recall", "precision", "r_precision", "map",
                             "hit_rate", "mrr", "ndcg")}
    base.update({f"{m}_ci": 0.05 for m in list(base)})
    base["n"] = 100
    base.update(over)
    return base


def ablation(**over):
    data = {
        "kind": "retrieval_ablation", "n_questions": 100, "embed_model": "bge-small",
        "chunk_max_tokens": 256,
        "a": [{"config": {"mode": m, "top_k": k}, "summary": summary()}
              for m in MODES for k in (1, 3, 5, 10, 20, 50)],
        "b": [{"config": {"mode": "hybrid", "top_k": k, "rerank": True,
                          "rerank_candidates": c, "min_rerank_score": None},
               "summary": summary(), "mean_returned": float(k)}
              for k in (1, 3, 5, 10) for c in (10, 40)],
        "w": [{"config": {"mode": m, "top_k": k, "keyword_weight": kw, "rrf_k": 60},
               "summary": summary()}
              for m in ("hybrid", "hybrid_bm25") for k in (5, 10, 20)
              for kw in (0.1, 0.5, 1.0)],
        "c": [{"config": {"arm": a, "mode": m, "top_k": k}, "summary": summary()}
              for m in ("semantic", "bm25") for k in (5, 10, 20)
              for a in ("none", "hyde", "multi_query", "multi_query+orig",
                        "decompose", "decompose+orig")],
        "timings": [{"config": {"mode": m, "top_k": k, "rerank": False}, "n": 100,
                     "latency_ms": {"embed": 20.0, "pinecone": 600.0, "hydrate": 3.0,
                                    "keyword_sql": 1300.0, "bm25": 0.5, "fuse": 0.1,
                                    "total": 900.0}}
                    for m in MODES for k in (1, 10)],
    }
    data.update(over)
    return data


def free_run():
    topics = ("agents", "alignment", "efficiency", "evaluation", "retrieval")
    synths = ("single_hop_specifc_query_synthesizer",
              "multi_hop_specific_query_synthesizer",
              "multi_hop_abstract_query_synthesizer")
    return {
        "kind": "retrieval_free", "n_questions": 100, "embed_model": "bge-small",
        "chunk_max_tokens": 256, "_source": "free-20260906T233943Z",
        "config": {"mode": "hybrid", "top_k": 10, "rerank": False},
        "overall": summary(ragas_precision=0.63, ragas_precision_ci=0.07,
                           ragas_recall=0.74, ragas_recall_ci=0.07,
                           precision_ceiling=0.163),
        "by_topic": {t: summary() for t in topics},
        "by_synthesizer": {s: summary() for s in synths},
        "per_question": [{"n_relevant": 1 + i % 3} for i in range(100)],
    }


def judged_run():
    metrics = ["faithfulness", "answer_relevancy",
               "llm_context_precision_with_reference", "context_recall"]
    return {
        "kind": "judged", "n_questions": 100, "n_abstentions": 3,
        "_source": "judged-20260907T015715Z",
        "means": {m: 0.8 for m in metrics},
        "means_excluding_abstentions": {m: 0.85 for m in metrics},
        "answerer_model": "qwen/qwen3-30b-a3b-instruct-2507",
        "judge_model": "deepseek/deepseek-v4-flash",
        "judge_spend": {"input_tokens": 1, "output_tokens": 1, "usd": 0.2294},
        "retrieval_config": {"mode": "hybrid", "top_k": 10, "rerank": False},
        "retrieval_metrics": summary(),
    }


def judge_scores():
    return {
        "model": "deepseek/deepseek-v4-flash",
        "primary": {"n": 81, "kappa": 0.828, "accuracy": 0.9136,
                    "tp": 35, "tn": 39, "fp": 7, "fn": 0},
        "sensitivity": {"n": 77, "kappa": 0.922, "accuracy": 0.961,
                        "tp": 35, "tn": 39, "fp": 3, "fn": 0},
        "by_case": {"supported": {"n": 29, "correct": 27},
                    "paraphrase": {"n": 10, "correct": 8},
                    "true_but_absent": {"n": 8, "correct": 5}},
    }


class TestVocabulary:
    def test_every_retrieval_mode_has_prose(self):
        """A legend showing `hybrid_bm25` is the defect this guards.

        `ts_rank` survives into the label on purpose - it is the name of the
        Postgres ranking function, not an identifier standing in for prose.
        """
        for mode in MODES:
            assert LABEL[mode] != mode
            assert "hybrid_" not in LABEL[mode]

    def test_every_query_arm_including_hyde_has_prose(self):
        for arm in ("none", "hyde", "multi_query", "multi_query+orig",
                    "decompose", "decompose+orig"):
            assert arm in LABEL

    def test_unknown_keys_degrade_to_readable_text(self):
        assert label("some_new_mode") == "some new mode"


class TestThemes:
    def test_both_themes_define_every_role(self):
        assert set(THEMES["light"]) == set(THEMES["dark"])

    def test_dark_is_selected_not_flipped(self):
        """Every role differs, and the ramps run opposite ways so the step
        nearest each surface still clears contrast."""
        light, dark = THEMES["light"], THEMES["dark"]
        assert light["surface"] != dark["surface"]
        assert light["series"][0] != dark["series"][0]
        assert light["ramp"][0] != dark["ramp"][0]

    def test_enough_series_slots_for_the_widest_figure(self):
        for t in THEMES.values():
            assert len(t["series"]) >= len(MODES)
            assert len(t["ramp"]) >= 6      # the six latency stages


class TestLoading:
    def write(self, tmp_path, name, blob, mtime):
        import os

        path = tmp_path / name
        path.write_text(json.dumps(blob))
        os.utime(path, (mtime, mtime))
        return path

    def test_newer_single_ablation_is_merged_over_the_full_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr("eval.figures.RESULTS", tmp_path)
        self.write(tmp_path, "ablation-all-20260907T005619Z.json", ablation(), 1000)
        only_c = {"kind": "retrieval_ablation", "c": [{"config": {"arm": "hyde"},
                                                       "summary": summary()}]}
        self.write(tmp_path, "ablation-c-20260907T020429Z.json", only_c, 2000)

        data = load_ablation()
        assert data["c"] == only_c["c"]          # the re-run won
        assert len(data["a"]) == 30              # everything else survived
        assert data["_source_c"] == "ablation-c-20260907T020429Z"

    def test_ordering_comes_from_the_filename_not_the_mtime(self, tmp_path, monkeypatch):
        """Copying results between checkouts rewrites every mtime to the same
        instant; the ISO stamp in the name is the only durable ordering."""
        monkeypatch.setattr("eval.figures.RESULTS", tmp_path)
        self.write(tmp_path, "ablation-all-20260907T005619Z.json", ablation(), 9000)
        newer = {"kind": "retrieval_ablation", "c": [{"config": {"arm": "hyde"},
                                                      "summary": summary()}]}
        self.write(tmp_path, "ablation-c-20260907T020429Z.json", newer, 1)

        assert load_ablation()["c"] == newer["c"]

    def test_provenance_names_the_run(self):
        line = provenance(free_run(), "free branch")
        assert "100 curated questions" in line
        assert "bge-small, 256-token chunks" in line
        assert "free-20260906T233943Z" in line


class TestRendering:
    """Everything below needs matplotlib, which is a `--with` extra."""

    @pytest.fixture(autouse=True)
    def _needs_matplotlib(self):
        pytest.importorskip("matplotlib")

    def test_a_subtitle_that_would_be_clipped_fails_loudly(self):
        from eval.figures import rc

        rc("light")
        with pytest.raises(AssertionError, match="over 104"):
            Frame("light", "t", "x" * 200, plot_h=2.0)

    def test_furniture_ignores_a_widened_axes_gutter(self):
        """Title and footer hang off a constant margin, so a dot plot's wide
        row-label gutter cannot shove them out of alignment - or off the
        right edge, which is how the footer got clipped."""
        from eval.figures import M_TEXT, rc

        rc("light")
        narrow = Frame("light", "t", "s", plot_h=2.0, m_left=0.86)
        wide = Frame("light", "t", "s", plot_h=2.0, m_left=2.15)
        xs = {round(f.fig.texts[0].get_position()[0], 6) for f in (narrow, wide)}
        assert len(xs) == 1
        assert xs.pop() == pytest.approx(M_TEXT / 8.6)

    def test_every_figure_renders_at_the_same_width_in_both_themes(self, tmp_path, monkeypatch):
        import eval.figures as figs

        monkeypatch.setattr(figs, "ASSETS", tmp_path)
        sources = {"ablation": ablation(), "free": free_run(),
                   "judged": judged_run(), "judge": judge_scores()}
        sources["ablation"]["_source"] = "ablation-all-20260907T005619Z"

        for name, (kind, fn) in figs.FIGURES.items():
            for theme in THEMES:
                figs.rc(theme)
                fig = fn(theme, *(sources[k] for k in kind.split("+")))
                assert fig.get_figwidth() == figs.FIG_W, name
                figs.save(fig, name, theme)

        for ext in FORMATS:
            written = {p.name for p in (tmp_path / ext).glob(f"*.{ext}")}
            assert len(written) == len(figs.FIGURES) * len(THEMES)

    def test_query_arms_plots_every_arm_that_was_measured(self):
        """HyDE was measured and silently dropped, because the figure carried
        its own hardcoded arm list."""
        import eval.figures as figs

        figs.rc("light")
        data = ablation()
        data["_source"] = "ablation-all-20260907T005619Z"
        fig = figs.fig_query_arms("light", data)
        rows = {t.get_text() for t in fig.axes[0].get_yticklabels()}
        assert rows == {LABEL[a] for a in ("none", "hyde", "multi_query",
                                           "multi_query+orig", "decompose",
                                           "decompose+orig")}

    def test_query_arms_marks_an_arm_that_was_never_run(self):
        """The real ablation only ran HyDE on the dense side. An empty row
        says so; dropping the row is what hid it."""
        import eval.figures as figs

        figs.rc("light")
        data = ablation()
        data["_source"] = "ablation-all-20260907T005619Z"
        data["c"] = [r for r in data["c"]
                     if not (r["config"]["mode"] == "bm25" and r["config"]["arm"] == "hyde")]
        fig = figs.fig_query_arms("light", data)
        assert any("not run" in t.get_text() for t in fig.axes[1].texts)

    def test_precision_panel_carries_the_arithmetic_ceiling(self):
        """Precision@k falls because most questions have one relevant chunk.
        Plotting the curve without the ceiling reads as a defect."""
        import eval.figures as figs

        figs.rc("light")
        data = ablation()
        data["_source"] = "ablation-all-20260907T005619Z"
        fig = figs.fig_retrieval_depth("light", data, free_run())
        dashed = [ln for ln in fig.axes[1].lines if ln.get_linestyle() != "-"]
        assert dashed, "no ceiling drawn"
        ceiling = dashed[0].get_ydata()
        assert ceiling[0] == pytest.approx(1.0)     # at k=1 it is 1.0 by definition
        assert ceiling[-1] < 0.1                    # and near zero at k=50
        assert list(ceiling) == sorted(ceiling, reverse=True)

    def test_the_judges_false_positives_are_not_painted_green(self):
        """A green bar on `waved through` - the hallucination the judge would
        miss - inverts the only reading that matters."""
        import eval.figures as figs

        for theme in THEMES:
            figs.rc(theme)
            fig = figs.fig_judge_agreement(theme, judge_scores())
            error_bar = fig.axes[1].patches[0]
            assert error_bar.get_facecolor()[:3] != pytest.approx((0.0, 0.514, 0.0), abs=0.01)
