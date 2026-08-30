"""Unit tests for eval/figures.py - pure SVG generation, no infra, no browser."""
import json

import pytest

from eval.figures import THEMES, Plot, card, fig_abstention, fig_keyword_fix, log_pos


def sweep():
    rows = []
    for mode in ("semantic", "keyword", "hybrid"):
        for k in (1, 5, 20):
            rows.append({
                "mode": mode, "top_k": k, "rerank": False, "rerank_top_k": None, "k": k,
                "recall": 0.5 + k / 100, "precision": 0.9 / k, "ndcg": 0.6,
                "mrr": 0.7, "hit_rate": 0.8, "abstention_precision": 0.0,
            })
    return {"results": rows}


class TestThemes:
    def test_both_themes_define_every_series(self):
        for t in THEMES.values():
            assert set(t["series"]) == {"semantic", "keyword", "hybrid"}

    def test_themes_are_distinct(self):
        """A dark figure on a light card is the bug this guards."""
        assert THEMES["light"]["card"] != THEMES["dark"]["card"]
        assert THEMES["light"]["ink"] != THEMES["dark"]["ink"]


class TestPlot:
    def t(self):
        return THEMES["light"]

    def test_end_labels_are_pushed_apart_when_curves_converge(self):
        """Three labels on the same pixel is worse than none - and the labels
        are the secondary encoding the palette's CVD margin relies on."""
        p = Plot(self.t(), 0, 0, 400, 200, "x", "y", [1, 2], log_pos([1, 2]))
        p.series({
            "semantic": [(1, 0.50), (2, 0.500)],
            "keyword": [(1, 0.50), (2, 0.501)],
            "hybrid": [(1, 0.50), (2, 0.502)],
        })
        ys = sorted(
            float(chunk.split('y="')[1].split('"')[0])
            for chunk in p.svg().split("<text")
            if 'font-weight="600"' in chunk
        )
        assert len(ys) == 3
        assert all(b - a >= 13 for a, b in zip(ys, ys[1:]))

    def test_labels_stay_inside_the_plot_box(self):
        p = Plot(self.t(), 0, 0, 400, 200, "x", "y", [1, 2], log_pos([1, 2]))
        p.series({m: [(1, 0.02), (2, 0.01)] for m in ("semantic", "keyword", "hybrid")})
        ys = [
            float(c.split('y="')[1].split('"')[0])
            for c in p.svg().split("<text") if 'font-weight="600"' in c
        ]
        assert max(ys) <= 200 + 4

    def test_fixed_label_column_overrides_the_anchor(self):
        p = Plot(self.t(), 0, 0, 400, 200, "x", "y", [1, 2], log_pos([1, 2]))
        p.series({"semantic": [(1, 0.5), (2, 0.6)]}, label_x=333)
        assert 'x="333.0"' in p.svg()

    def test_y_axis_maps_zero_to_the_baseline(self):
        p = Plot(self.t(), 0, 10, 400, 200, "x", "y", [1], log_pos([1, 2]))
        assert p.sy(0.0) == 210
        assert p.sy(1.0) == 10


class TestCard:
    def test_emits_standalone_svg_with_namespace(self):
        svg = card(THEMES["light"], 400, 300, "T", "S", "<g/>")
        assert svg.startswith("<svg xmlns=")
        assert svg.rstrip().endswith("</svg>")

    def test_escapes_text(self):
        svg = card(THEMES["light"], 400, 300, "a < b & c", "S", "")
        assert "a &lt; b &amp; c" in svg

    def test_legend_sits_below_the_subtitle_not_on_the_axis(self):
        """At the foot of the card it collided with the x-axis title."""
        svg = card(THEMES["light"], 400, 300, "T", "S", "", legend=[("x", "#000")])
        y = float(svg.split('<rect x="24" y="')[1].split('"')[0])
        assert y < 100


class TestFigures:
    def test_keyword_fix_renders_both_series(self):
        svg = fig_keyword_fix(THEMES["dark"], sweep(), sweep())
        assert "before" in svg and "after" in svg

    def test_abstention_marks_the_shipped_default(self):
        data = {"results": [
            {"min_rerank_score": f, "recall": 0.8, "ndcg": 0.8,
             "abstention_precision": 0.0 if f < -8 else 1.0}
            for f in (-12, -8, -4, 0)
        ]}
        svg = fig_abstention(THEMES["light"], data)
        assert "default" in svg
        assert "stroke-dasharray" in svg  # the marker line

    def test_figures_differ_between_themes(self):
        light = fig_keyword_fix(THEMES["light"], sweep(), sweep())
        dark = fig_keyword_fix(THEMES["dark"], sweep(), sweep())
        assert light != dark
        assert THEMES["dark"]["card"] in dark
