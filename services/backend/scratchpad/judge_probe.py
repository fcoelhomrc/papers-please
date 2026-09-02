"""Does a candidate judge actually judge correctly?

Three cases with known-correct verdicts. A usable judge must (a) complete
without parse errors, and (b) rank supported > partly-fabricated >
unsupported. A cheap judge that can't do (b) is worth nothing.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness
from config import load
from orchestrator.llm import openrouter_chat

CTX = ["A single RL recovery policy recovers a quadruped from 94% of sampled "
       "fallen postures on flat ground within 3.2 seconds on average, compared "
       "with 61% for the scripted baseline."]

CASES = [
    ("supported",  "The learned policy recovers from 94% of fallen postures, against 61% for the scripted baseline."),
    ("fabricated", "The learned policy recovers from 94% of postures, was trained for 200 GPU-hours on 8 A100s, and was deployed commercially in 2024."),
    ("unsupported","The paper shows transformers outperform LSTMs on speech recognition benchmarks."),
]

cfg = load()
for model in sys.argv[1:]:
    try:
        judge = LangchainLLMWrapper(openrouter_chat(model, 2048, cfg))
        ds = EvaluationDataset.from_list([
            {"user_input": "What does the recovery policy achieve?",
             "response": ans, "retrieved_contexts": CTX, "reference": "94% recovery rate."}
            for _, ans in CASES
        ])
        t0 = time.time()
        df = evaluate(ds, metrics=[faithfulness], llm=judge, show_progress=False).to_pandas()
        scores = [round(float(v), 2) for v in df["faithfulness"]]
        ok = scores[0] > scores[1] > scores[2] or (scores[0] > scores[1] and scores[0] > scores[2])
        nan = any(s != s for s in scores)
        verdict = "PARSE/NaN" if nan else ("discriminates" if ok else "DOES NOT discriminate")
        print(f"{model:<42} {time.time()-t0:5.1f}s  supported={scores[0]:<5} fabricated={scores[1]:<5} unsupported={scores[2]:<5}  {verdict}")
    except Exception as e:
        print(f"{model:<42} FAILED {type(e).__name__}: {str(e)[:70]}")
