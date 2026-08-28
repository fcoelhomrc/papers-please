"""Fixed, real content the eval dataset's questions are grounded in -
independent of whatever happens to be in the dev DB at eval time (manually
fetched papers, wiped/reset state, etc.). Real abstracts pulled from papers
genuinely in Semantic Scholar (not synthetic filler), given distinct
source_ids so they never collide with anything actually fetched through
the normal pipeline and are trivially identifiable/removable.
"""

FIXTURES = [
    {
        "source_id": "eval-fixture-icub-promp",
        "title": "Prediction of Intention during Interaction with iCub with Probabilistic Movement Primitives",
        "chunks": [
            "This paper describes our open-source software for predicting the "
            "intention of a user physically interacting with the humanoid robot "
            "iCub. Our goal is to allow the robot to infer the intention of the "
            "human partner during collaboration, by predicting the future intended "
            "trajectory: this capability is critical to design anticipatory "
            "behaviors that are crucial in human-robot collaborative scenarios."
        ],
    },
    {
        "source_id": "eval-fixture-muscle-synergy-rl",
        "title": "Low-Rank Modular Reinforcement Learning via Muscle Synergy",
        "chunks": [
            "Modular Reinforcement Learning (RL) decentralizes the control of "
            "multi-joint robots by learning policies for each actuator. Previous "
            "work on modular RL has proven its ability to control morphologically "
            "different agents with a shared actuator policy. However, with the "
            "increase in the Degree of Freedom (DoF) of robots, training a "
            "morphology-generalizable modular controller becomes exponentially "
            "difficult."
        ],
    },
    {
        "source_id": "eval-fixture-fall-recovery-wheeled",
        "title": "Robust Fall Recovery for Armless Bipedal-Wheeled Robots via Force-Guided Learning",
        "chunks": [
            "Fall recovery is critical for autonomous legged locomotion. Existing "
            "methods have demonstrated that some legged robots, such as humanoids "
            "and quadrupeds, are capable of fall recovery from diverse postures by "
            "utilizing arms or coordinating multi-legs to generate support forces. "
            "Without arms or other legs to provide supportive assistance, a "
            "bipedal-wheeled robot must rely solely on the actuation of its wheels, "
            "guided by force-based learning."
        ],
    },
    {
        "source_id": "eval-fixture-supervisory-control-review",
        "title": "A Review of Supervisory Control Strategies for Walking Robots",
        "chunks": [
            "This review synthesizes advances in supervisory control for walking "
            "robots, integrating perspectives on architectural frameworks and "
            "decision-making strategies, and analyzing their performance across "
            "diverse application contexts. We survey centralized, decentralized or "
            "distributed, hierarchical, and hybrid architectures, then examine "
            "rule-based, model-based, AI-driven, fuzzy, event-driven, and adaptive "
            "decision-making strategies."
        ],
    },
]
