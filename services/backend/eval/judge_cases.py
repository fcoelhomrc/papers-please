"""Labelled (context, statement) pairs for comparing LLM judges.

Ragas' faithfulness metric reduces to one binary call: given a context and a
statement extracted from an answer, is the statement supported? That is the
step where judges actually differ, so this is what gets measured - directly,
without the statement-extraction step in between, which would otherwise add
variance that has nothing to do with the judge.

Every case carries a proposed label, derived by construction rather than by
opinion: a statement asserting a number the context does not contain is
unsupported, and that is arithmetic, not a judgement call. The labelling UI
(eval/labelling/index.html) exists so a human confirms or overrides them,
which turns "labels by construction" into "labels by construction, verified"
- a defensible ground truth rather than my say-so.

CONTEXTS ARE REAL. They are abstracts from eval/fixtures.py, the same text
the pipeline actually indexes. Synthetic contexts would make the task
cleaner than the job the judge really does.

The case types are chosen so that most of them *discriminate*. Four models
already scored identically on obvious supported/fabricated pairs, which is
why that probe decided nothing. The cases that separate judges are the ones
where a statement is nearly right:

    corrupted_number   a figure altered - 94% -> 84%
    corrupted_entity   the wrong subject - quadruped -> humanoid
    overgeneralised    a scoped claim stated unconditionally
    unsupported_cause  a causal "because" the context never gives
    true_but_absent    factually true in the world, absent from the context
    paraphrase         faithful, reworded - catches over-strict judges

`true_but_absent` is the sharpest of them. Faithfulness asks whether the
context supports the statement, not whether the statement is true, and a
judge that answers the second question is measuring the wrong thing while
looking confident.
"""
import json
from pathlib import Path

# --- contexts, verbatim from eval/fixtures.py -----------------------------

ICUB = (
    "This paper describes our open-source software for predicting the intention "
    "of a user physically interacting with the humanoid robot iCub. Our goal is "
    "to allow the robot to infer the intention of the human partner during "
    "collaboration, by predicting the future intended trajectory: this "
    "capability is critical to design anticipatory behaviors that are crucial "
    "in human-robot collaborative scenarios."
)

MODULAR_RL = (
    "Modular Reinforcement Learning (RL) decentralizes the control of "
    "multi-joint robots by learning policies for each actuator. Previous work "
    "on modular RL has proven its ability to control morphologically different "
    "agents with a shared actuator policy. However, with the increase in the "
    "Degree of Freedom (DoF) of robots, training a morphologically diverse "
    "population becomes increasingly difficult."
)

FALL_RECOVERY = (
    "Fall recovery is critical for autonomous legged locomotion. Existing "
    "methods have demonstrated that some legged robots, such as humanoids and "
    "quadrupeds, are capable of fall recovery from diverse postures by "
    "utilizing arms or coordinating multi-legs to generate support forces. "
    "Without arms or other legs to provide supportive assistance, a wheeled "
    "biped presents a harder case."
)

SUPERVISORY = (
    "This review synthesizes advances in supervisory control for walking "
    "robots, integrating perspectives on architectural frameworks and "
    "decision-making strategies, and analyzing their performance across "
    "diverse application contexts. We survey centralized, decentralized or "
    "distributed, hierarchical, and hybrid architectures."
)

POLLUTANT = (
    "Air pollution poses a critical global public health challenge. "
    "Molecular-level initiating events, such as pollutant-protein "
    "interactions, can trigger cascades of biological responses that may "
    "contribute to adverse health effects. However, current methods are "
    "limited in their ability to systematically identify these early binding "
    "events."
)

SKIN = (
    "Skin diseases are among the most common health conditions, and early "
    "detection is crucial for effective treatment and prevention of serious "
    "complications. However, traditional diagnostic methods require expert "
    "consultation and specialized equipment, which may not be easily "
    "accessible to everyone."
)

# A results-style context, so numeric corruption has something to bite on.
RECOVERY_RESULTS = (
    "Results. We train a single recovery policy in simulation and deploy it "
    "without modification on hardware. The policy recovers the robot from 94% "
    "of sampled fallen postures on flat ground within 3.2 seconds on average, "
    "compared with 61% for the scripted baseline, which requires a "
    "hand-authored routine per posture class."
)

# (context, statement, proposed_label, case_type)
# label 1 = supported by the context, 0 = not supported
CASES: list[tuple[str, str, int, str]] = [
    # --- fully supported: catches over-strict judges -----------------------
    (ICUB, "The software predicts the intention of a user interacting with iCub.", 1, "supported"),
    (ICUB, "iCub is a humanoid robot.", 1, "supported"),
    (ICUB, "Predicting the intended trajectory supports anticipatory behaviour.", 1, "supported"),
    (RECOVERY_RESULTS, "The policy recovers the robot from 94% of sampled fallen postures.", 1, "supported"),
    (RECOVERY_RESULTS, "The scripted baseline recovers from 61% of postures.", 1, "supported"),
    (FALL_RECOVERY, "Humanoids and quadrupeds can recover from diverse postures.", 1, "supported"),
    (MODULAR_RL, "Modular RL learns a policy for each actuator.", 1, "supported"),
    (SUPERVISORY, "The review surveys hierarchical architectures.", 1, "supported"),

    # --- faithful paraphrase: same claim, different words ------------------
    (ICUB, "The system forecasts where the human partner intends to move next.", 1, "paraphrase"),
    (RECOVERY_RESULTS, "Recovery succeeded in roughly nineteen of every twenty attempts.", 1, "paraphrase"),
    (MODULAR_RL, "Control is decentralised rather than handled by one central policy.", 1, "paraphrase"),
    (SKIN, "Catching skin conditions early matters for treating them well.", 1, "paraphrase"),
    (POLLUTANT, "Interactions between pollutants and proteins can set off downstream biological effects.", 1, "paraphrase"),

    # --- corrupted number: the single most realistic RAG failure -----------
    (RECOVERY_RESULTS, "The policy recovers the robot from 84% of sampled fallen postures.", 0, "corrupted_number"),
    (RECOVERY_RESULTS, "The scripted baseline recovers from 71% of postures.", 0, "corrupted_number"),
    (RECOVERY_RESULTS, "Recovery takes 1.2 seconds on average.", 0, "corrupted_number"),
    (RECOVERY_RESULTS, "The policy was evaluated on 94 distinct robots.", 0, "corrupted_number"),

    # --- corrupted entity: right shape, wrong subject ----------------------
    (ICUB, "The software predicts the intention of a user interacting with the Atlas robot.", 0, "corrupted_entity"),
    (RECOVERY_RESULTS, "The policy was trained on hardware and deployed in simulation.", 0, "corrupted_entity"),
    (FALL_RECOVERY, "Wheeled bipeds recover more easily than quadrupeds.", 0, "corrupted_entity"),
    (MODULAR_RL, "Modular RL learns a single policy shared across all joints of one robot.", 0, "corrupted_entity"),

    # --- overgeneralised: scope silently dropped ---------------------------
    (RECOVERY_RESULTS, "The policy recovers the robot from any fallen posture on any terrain.", 0, "overgeneralised"),
    (FALL_RECOVERY, "All legged robots can recover from a fall.", 0, "overgeneralised"),
    (SUPERVISORY, "Hierarchical architectures outperform all other architectures for walking robots.", 0, "overgeneralised"),
    (SKIN, "Traditional diagnostic methods are inaccessible to everyone.", 0, "overgeneralised"),

    # --- unsupported cause: a "because" the context never gives ------------
    (RECOVERY_RESULTS, "The policy outperforms the baseline because it uses a larger neural network.", 0, "unsupported_cause"),
    (MODULAR_RL, "Training gets harder with more DoF because the reward signal becomes sparse.", 0, "unsupported_cause"),
    (POLLUTANT, "Air pollution is a public health challenge because of rising vehicle ownership.", 0, "unsupported_cause"),

    # --- true but absent: the sharpest discriminator -----------------------
    # Each is true in the world and unsupported by THIS context. A judge that
    # rewards world-knowledge is answering the wrong question.
    (ICUB, "iCub was developed by the Italian Institute of Technology.", 0, "true_but_absent"),
    (MODULAR_RL, "Reinforcement learning agents learn from a reward signal.", 0, "true_but_absent"),
    (SKIN, "Convolutional neural networks are widely used for image classification.", 0, "true_but_absent"),
    (FALL_RECOVERY, "Quadruped robots typically have four legs.", 0, "true_but_absent"),
    (POLLUTANT, "Particulate matter is a component of air pollution.", 0, "true_but_absent"),

    # --- wholly unrelated: the floor --------------------------------------
    (ICUB, "Transformers outperform LSTMs on speech recognition benchmarks.", 0, "unrelated"),
    (RECOVERY_RESULTS, "The study analysed protein folding in aqueous solution.", 0, "unrelated"),
    (SKIN, "The paper introduces a new consensus algorithm for distributed databases.", 0, "unrelated"),

    # --- negation: reverses the context's claim ----------------------------
    (FALL_RECOVERY, "Fall recovery is not important for autonomous legged locomotion.", 0, "negation"),
    (SKIN, "Early detection of skin disease has little effect on outcomes.", 0, "negation"),
    (MODULAR_RL, "Increasing the degrees of freedom makes training easier.", 0, "negation"),

    # ---------------------------------------------------------------------
    # Second block. Added to reach ~80 statements and to rebalance the
    # classes: at 13 supported against 26 unsupported, a judge could score
    # respectably by answering "unsupported" to everything, and kappa gets
    # unstable when one class dominates. The split below is ~45/55.
    # ---------------------------------------------------------------------

    # --- supported --------------------------------------------------------
    (ICUB, "The software is open source.", 1, "supported"),
    (ICUB, "The robot infers the human partner's intention during collaboration.", 1, "supported"),
    (ICUB, "Anticipatory behaviours matter in human-robot collaboration.", 1, "supported"),
    (MODULAR_RL, "A shared actuator policy can control morphologically different agents.", 1, "supported"),
    (MODULAR_RL, "Training becomes harder as the degrees of freedom increase.", 1, "supported"),
    (MODULAR_RL, "Modular RL decentralises control of multi-joint robots.", 1, "supported"),
    (FALL_RECOVERY, "Some legged robots recover using their arms.", 1, "supported"),
    (FALL_RECOVERY, "A wheeled biped lacks arms or extra legs for support.", 1, "supported"),
    (FALL_RECOVERY, "Fall recovery matters for autonomous legged locomotion.", 1, "supported"),
    (SUPERVISORY, "The review covers centralized and decentralized architectures.", 1, "supported"),
    (SUPERVISORY, "The paper analyses performance across different application contexts.", 1, "supported"),
    (SUPERVISORY, "Hybrid architectures are among those surveyed.", 1, "supported"),
    (POLLUTANT, "Pollutant-protein interactions are molecular-level initiating events.", 1, "supported"),
    (POLLUTANT, "Existing methods struggle to identify early binding events systematically.", 1, "supported"),
    (SKIN, "Skin diseases are among the most common health conditions.", 1, "supported"),
    (SKIN, "Traditional diagnosis needs expert consultation and specialised equipment.", 1, "supported"),
    (RECOVERY_RESULTS, "The policy was trained in simulation.", 1, "supported"),
    (RECOVERY_RESULTS, "The policy was deployed on hardware without modification.", 1, "supported"),
    (RECOVERY_RESULTS, "Recovery takes 3.2 seconds on average.", 1, "supported"),
    (RECOVERY_RESULTS, "The scripted baseline needs a hand-authored routine per posture class.", 1, "supported"),
    (RECOVERY_RESULTS, "Testing was carried out on flat ground.", 1, "supported"),

    # --- faithful paraphrase ----------------------------------------------
    (FALL_RECOVERY, "Getting back up unaided is a harder problem for a robot on wheels.", 1, "paraphrase"),
    (SUPERVISORY, "The authors compare several ways of organising a control hierarchy.", 1, "paraphrase"),
    (RECOVERY_RESULTS, "The learned controller beat the hand-written one by a wide margin.", 1, "paraphrase"),
    (POLLUTANT, "Dirty air is a serious worldwide health problem.", 1, "paraphrase"),
    (MODULAR_RL, "Scaling to robots with many joints complicates training.", 1, "paraphrase"),
    (ICUB, "Guessing the partner's next movement lets the robot act in advance.", 1, "paraphrase"),

    # --- corrupted number -------------------------------------------------
    (RECOVERY_RESULTS, "Recovery takes 32 seconds on average.", 0, "corrupted_number"),
    (RECOVERY_RESULTS, "The policy improves on the baseline by 12 percentage points.", 0, "corrupted_number"),

    # --- corrupted entity -------------------------------------------------
    (SUPERVISORY, "The review is limited to centralized architectures.", 0, "corrupted_entity"),
    (POLLUTANT, "The work identifies interactions between pollutants and DNA.", 0, "corrupted_entity"),
    (SKIN, "The paper addresses diagnosis of cardiovascular disease.", 0, "corrupted_entity"),

    # --- overgeneralised --------------------------------------------------
    (MODULAR_RL, "Modular RL solves control for robots of any morphology.", 0, "overgeneralised"),
    (POLLUTANT, "Every pollutant-protein interaction leads to adverse health effects.", 0, "overgeneralised"),

    # --- unsupported cause ------------------------------------------------
    (ICUB, "The robot predicts intention because it tracks the human's gaze.", 0, "unsupported_cause"),
    (SKIN, "Diagnostic equipment is inaccessible because of regulatory barriers.", 0, "unsupported_cause"),
    (FALL_RECOVERY, "Wheeled bipeds are harder to recover because their wheels lose traction.", 0, "unsupported_cause"),

    # --- true but absent --------------------------------------------------
    (SUPERVISORY, "Walking robots must maintain balance to avoid falling.", 0, "true_but_absent"),
    (RECOVERY_RESULTS, "Simulation-to-real transfer is a known challenge in robotics.", 0, "true_but_absent"),
    (POLLUTANT, "Proteins are made up of amino acids.", 0, "true_but_absent"),

    # --- negation ---------------------------------------------------------
    (ICUB, "The software cannot predict the user's future trajectory.", 0, "negation"),
    (RECOVERY_RESULTS, "The scripted baseline outperformed the learned policy.", 0, "negation"),

    # --- unrelated --------------------------------------------------------
    (MODULAR_RL, "The authors report state-of-the-art results on ImageNet.", 0, "unrelated"),
    (SUPERVISORY, "The review focuses on federated learning privacy guarantees.", 0, "unrelated"),
]


def as_records() -> list[dict]:
    """Cases as dicts, with a stable id so labels survive reordering."""
    return [
        {
            "id": f"c{i:03d}",
            "context": context,
            "statement": statement,
            "proposed_label": label,
            "case_type": case_type,
        }
        for i, (context, statement, label, case_type) in enumerate(CASES)
    ]


def summary() -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, _, _, case_type in CASES:
        counts[case_type] = counts.get(case_type, 0) + 1
    return counts


LABELS_PATH = Path(__file__).parent / "labelling" / "labels.json"


def load_labels(path: Path | None = None) -> dict[str, int]:
    """Human labels exported from the labelling page, keyed by case id.

    `null` means the labeller marked it Unsure. Those are dropped rather
    than coerced: an item nobody could decide is not ground truth, and
    guessing at it would put noise into the very numbers this exists to
    make trustworthy. Every downstream metric is computed over the labels
    that survive.
    """
    path = path or LABELS_PATH
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {
        row["id"]: int(row["label"])
        for row in data.get("labels", [])
        if row.get("label") is not None
    }


def agreement_with_proposed(labels: dict[str, int]) -> dict:
    """How far the human labels moved my constructed ones.

    Reported rather than hidden. A high disagreement rate does not mean the
    labeller was wrong - it means the cases were sloppier than claimed, and
    the honest response is to fix the cases, not to quietly keep the
    labels I preferred.
    """
    proposed = {r["id"]: r["proposed_label"] for r in as_records()}
    shared = [cid for cid in labels if cid in proposed]
    disagreed = [cid for cid in shared if labels[cid] != proposed[cid]]
    by_type = {r["id"]: r["case_type"] for r in as_records()}
    return {
        "n_labelled": len(shared),
        "n_disagreed": len(disagreed),
        "rate": (len(disagreed) / len(shared)) if shared else 0.0,
        "disagreed": [{"id": c, "case_type": by_type[c]} for c in disagreed],
    }


if __name__ == "__main__":
    supported = sum(1 for *_, label, _ in [(c) for c in CASES] if label == 1)
    print(f"{len(CASES)} statements across {len(summary())} case types")
    print(f"  supported: {supported}  unsupported: {len(CASES) - supported}")
    for case_type, n in sorted(summary().items(), key=lambda kv: -kv[1]):
        print(f"  {case_type:<20} {n}")
