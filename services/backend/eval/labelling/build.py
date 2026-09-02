"""Generates the standalone statement-labelling page.

    uv run python -m eval.labelling.build

Self-contained HTML with the cases baked in. Opened as a `file://` URL there
is no server to fetch from, and a labelling tool that needs a dev server
running is a labelling tool that doesn't get used.

Two fields are deliberately withheld from the page:

  proposed_label  showing my construction would anchor the labeller to it,
                  and independent confirmation is the entire point
  case_type       "corrupted_number" answers the question by itself

Both are re-joined by `id` at analysis time from eval/judge_cases.py.
"""
import json
from pathlib import Path

from eval.judge_cases import as_records

HERE = Path(__file__).parent
HIDDEN = ("proposed_label", "case_type")

TEMPLATE = """<!doctype html>
<html lang="en" data-theme="auto">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Faithfulness labelling — Papers Please</title>
<style>
  :root {
    --bg:#fcfcfb; --surface:#fff; --inset:#f4f4f2; --border:#e7e5e4;
    --ink:#0b0b0b; --muted:#57534e; --faint:#8b8781;
    --yes:#15803d; --yes-soft:#dcfce7; --no:#b91c1c; --no-soft:#fee2e2;
    --accent:#4338ca; --accent-soft:#e0e7ff;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg:#1a1a19; --surface:#232322; --inset:#2b2b29; --border:#3a3a37;
      --ink:#f5f5f4; --muted:#c3c2b7; --faint:#8b8781;
      --yes:#4ade80; --yes-soft:#14532d; --no:#f87171; --no-soft:#601b1b;
      --accent:#a5b4fc; --accent-soft:#312e81;
    }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Helvetica,sans-serif; }
  .wrap { max-width:820px; margin:0 auto; padding:24px 20px 96px; }
  h1 { font-size:20px; margin:0 0 2px; letter-spacing:-0.01em; }
  .sub { color:var(--muted); font-size:14px; margin:0 0 18px; }

  .bar { height:6px; background:var(--inset); border-radius:99px; overflow:hidden; margin:14px 0 6px; }
  .bar > div { height:100%; background:var(--accent); transition:width .2s; }
  .counts { display:flex; gap:14px; font-size:12.5px; color:var(--faint);
    font-variant-numeric:tabular-nums; margin-bottom:20px; flex-wrap:wrap; }

  details.help { background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:12px 14px; margin-bottom:20px; }
  details.help summary { cursor:pointer; font-weight:600; font-size:14px; }
  details.help ul { margin:10px 0 0; padding-left:20px; font-size:14px; color:var(--muted); }
  details.help li { margin-bottom:7px; }
  details.help b { color:var(--ink); }
  .rule { background:var(--accent-soft); color:var(--ink); border-radius:8px;
    padding:10px 12px; font-size:14px; margin:12px 0 0; }

  .card { background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:18px; }
  .label { font-size:11px; text-transform:uppercase; letter-spacing:.06em;
    color:var(--faint); margin-bottom:6px; }
  .ctx { background:var(--inset); border-radius:8px; padding:12px 14px;
    font-size:14.5px; color:var(--muted); max-height:220px; overflow:auto;
    white-space:pre-wrap; }
  .stmt { font-size:18px; line-height:1.5; margin:8px 0 0; padding:14px;
    border-left:3px solid var(--accent); background:var(--inset); border-radius:0 8px 8px 0; }

  .btns { display:flex; gap:10px; margin-top:18px; flex-wrap:wrap; }
  button { font:inherit; font-size:15px; font-weight:600; cursor:pointer;
    border-radius:9px; padding:11px 18px; border:1px solid var(--border);
    background:var(--surface); color:var(--ink); transition:.12s; }
  button:hover { transform:translateY(-1px); }
  button kbd { font:inherit; font-size:11px; opacity:.6; margin-left:7px; }
  .yes { background:var(--yes-soft); border-color:transparent; color:var(--yes); }
  .no  { background:var(--no-soft);  border-color:transparent; color:var(--no); }
  .ghost { color:var(--muted); }
  .nav { margin-left:auto; display:flex; gap:8px; }

  .done { text-align:center; padding:40px 20px; }
  .done h2 { margin:0 0 8px; }
  textarea { width:100%; height:220px; margin-top:14px; font:12px/1.5 ui-monospace,monospace;
    background:var(--inset); color:var(--ink); border:1px solid var(--border);
    border-radius:8px; padding:12px; }
  .row { display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin-top:14px; }
  .ok { color:var(--yes); font-size:13px; min-height:18px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Faithfulness labelling</h1>
  <p class="sub">Is each statement supported by the passage above it? __N__ statements, roughly 30&nbsp;minutes. Progress saves automatically — you can close the tab and come back.</p>

  <details class="help" open>
    <summary>How to decide</summary>
    <ul>
      <li><b>Judge only against the passage shown.</b> Not against what you know to be true, and not against the paper as a whole.</li>
      <li>Ask: <b>is every part of this statement stated or directly implied by that passage?</b> If any part isn't — a number, a name, a cause, a scope — it is <b>not supported</b>.</li>
      <li>A <b>true</b> statement that the passage simply doesn't mention is <b>not supported</b>. This is the case people get wrong most often, and it is deliberately common here.</li>
      <li>Reworded but same meaning → <b>supported</b>. Don't demand matching vocabulary.</li>
      <li>A statement that adds a reason ("because …") the passage never gives → <b>not supported</b>.</li>
      <li>A statement that drops a qualifier — "on flat ground", "some robots" — and claims it generally → <b>not supported</b>.</li>
      <li>Use <b>Unsure</b> freely. Skipped items are excluded from scoring rather than guessed at, which is better than a coin flip.</li>
    </ul>
    <p class="rule">The question is <b>“does the passage support this?”</b> — never “is this true?”</p>
  </details>

  <div class="bar"><div id="fill" style="width:0%"></div></div>
  <div class="counts">
    <span id="pos">0 / __N__</span><span id="tally"></span><span id="saved" class="ok"></span>
  </div>

  <div id="stage"></div>
</div>

<script>
const CASES = __CASES__;
const KEY = "pp-faithfulness-labels-v1";
let labels = {};
let i = 0;

try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { labels = {}; }
// Resume at the first unlabelled case rather than restarting from the top.
i = CASES.findIndex(c => !(c.id in labels));
if (i < 0) i = CASES.length;

const el = id => document.getElementById(id);

function save() {
  try {
    localStorage.setItem(KEY, JSON.stringify(labels));
    el("saved").textContent = "saved";
    setTimeout(() => { el("saved").textContent = ""; }, 900);
  } catch (e) {
    el("saved").textContent = "could not save locally — export before closing";
  }
}

function set(value) {
  if (i >= CASES.length) return;
  labels[CASES[i].id] = value;
  save();
  i++;
  render();
}

function back() { if (i > 0) { i--; render(); } }

function render() {
  const done = Object.keys(labels).length;
  el("fill").style.width = (100 * done / CASES.length) + "%";
  el("pos").textContent = done + " / " + CASES.length;
  const yes = Object.values(labels).filter(v => v === 1).length;
  const no = Object.values(labels).filter(v => v === 0).length;
  const unsure = Object.values(labels).filter(v => v === null).length;
  el("tally").textContent = `${yes} supported · ${no} not · ${unsure} unsure`;

  if (i >= CASES.length) return finish();

  const c = CASES[i];
  const prev = labels[c.id];
  el("stage").innerHTML = `
    <div class="card">
      <div class="label">Passage</div>
      <div class="ctx">${esc(c.context)}</div>
      <div class="label" style="margin-top:16px">Statement</div>
      <div class="stmt">${esc(c.statement)}</div>
      <div class="btns">
        <button class="yes" onclick="set(1)">Supported<kbd>S</kbd></button>
        <button class="no" onclick="set(0)">Not supported<kbd>N</kbd></button>
        <button class="ghost" onclick="set(null)">Unsure<kbd>U</kbd></button>
        <span class="nav">
          <button class="ghost" onclick="back()" ${i === 0 ? "disabled" : ""}>← Back</button>
        </span>
      </div>
      ${prev !== undefined ? `<p class="sub" style="margin:12px 0 0">Previously: <b>${prev === 1 ? "supported" : prev === 0 ? "not supported" : "unsure"}</b></p>` : ""}
    </div>`;
}

function finish() {
  const out = CASES.map(c => ({ id: c.id, label: labels[c.id] ?? null }));
  const json = JSON.stringify({ labelled_at: new Date().toISOString(), labels: out }, null, 1);
  el("stage").innerHTML = `
    <div class="card done">
      <h2>All ${CASES.length} labelled</h2>
      <p class="sub">Save the file into <code>services/backend/eval/labelling/</code>.</p>
      <div class="row">
        <button class="yes" onclick="download()">Download labels.json</button>
        <button onclick="copy()">Copy to clipboard</button>
        <button class="ghost" onclick="back()">← Review last</button>
      </div>
      <p class="ok" id="msg"></p>
      <textarea id="json" readonly>${esc(json)}</textarea>
    </div>`;
}

function download() {
  const blob = new Blob([el("json").value], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels.json";
  a.click();
  URL.revokeObjectURL(a.href);
  el("msg").textContent = "downloaded";
}

async function copy() {
  const text = el("json").value;
  try {
    await navigator.clipboard.writeText(text);
    el("msg").textContent = "copied";
  } catch (e) {
    // Clipboard access is restricted in some file:// contexts; selecting the
    // textarea always works as a fallback.
    el("json").select();
    el("msg").textContent = "press Ctrl/Cmd-C to copy the selected text";
  }
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
}

document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  const k = e.key.toLowerCase();
  if (k === "s" || k === "1") { e.preventDefault(); set(1); }
  else if (k === "n" || k === "0") { e.preventDefault(); set(0); }
  else if (k === "u") { e.preventDefault(); set(null); }
  else if (k === "arrowleft" || k === "backspace") { e.preventDefault(); back(); }
});

render();
</script>
</body>
</html>
"""


def build() -> Path:
    records = [
        {k: v for k, v in r.items() if k not in HIDDEN} for r in as_records()
    ]
    html = (
        TEMPLATE
        .replace("__CASES__", json.dumps(records))
        .replace("__N__", str(len(records)))
    )
    out = HERE / "index.html"
    out.write_text(html)
    return out


if __name__ == "__main__":
    path = build()
    print(f"{path}  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"open it with:  xdg-open {path}")
