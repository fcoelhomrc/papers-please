"""Fixed, real content the eval dataset's questions are grounded in -
independent of whatever happens to be in the dev DB at eval time (manually
fetched papers, wiped/reset state, etc.). Real abstracts pulled live from
the Semantic Scholar API (the same one this app fetches from - not
memory-reconstructed, not synthetic filler), given distinct source_ids so
they never collide with anything actually fetched through the normal
pipeline and are trivially identifiable/removable.

Deliberately spans distinct domains, not just robotics (the domain that
happened to dominate the dev DB from earlier manual testing) - an eval set
that only ever tests one topic can't tell you whether retrieval is
actually discriminating between subjects or just returning "the biggest
document" regardless of relevance.
"""

FIXTURES = [
    # --- robotics / control (original 4) ---
    {
        "source_id": "eval-fixture-icub-promp",
        "title": "Prediction of Intention during Interaction with iCub with Probabilistic Movement Primitives",
        "domain": "robotics",
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
        "domain": "robotics",
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
        "domain": "robotics",
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
        "domain": "robotics",
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
    # --- other domains, pulled live from Semantic Scholar for this expansion ---
    {
        "source_id": "eval-fixture-tipformer-pollutant-protein",
        "title": "A Transformer-Based Deep Learning Approach to Predicting Air Organic Pollutant-Human Protein Interactions",
        "domain": "toxicology / bioinformatics",
        "chunks": [
            "Air pollution poses a critical global public health challenge. "
            "Molecular-level initiating events, such as pollutant-protein "
            "interactions, can trigger cascades of biological responses that may "
            "contribute to adverse health effects. However, current methods are "
            "limited in their ability to systematically identify these early "
            "binding events, particularly for emerging airborne pollutants, which "
            "hinders mechanistic understanding and risk assessment of "
            "pollution-related toxicity. To address this, we developed tipFormer "
            "(pollutant-protein interaction prediction based on transformer), a "
            "novel deep learning approach for predicting interactions between "
            "airborne organic pollutants and human proteins."
        ],
    },
    {
        "source_id": "eval-fixture-skin-disease-cnn",
        "title": "Early Screening and Detection of Skin Diseases using Deep Learning",
        "domain": "medical imaging",
        "chunks": [
            "Skin diseases are among the most common health conditions, and early "
            "detection is crucial for effective treatment and prevention of "
            "serious complications. However, traditional diagnostic methods "
            "require expert consultation and specialized equipment, which may not "
            "be easily accessible to everyone. To address this challenge, this "
            "paper presents an AI-based system for early screening and detection "
            "of skin diseases using deep learning techniques. The proposed system "
            "utilizes a Convolutional Neural Network (CNN) model to classify skin "
            "images into three categories: melanoma, benign, and normal."
        ],
    },
    {
        "source_id": "eval-fixture-grain-yield-forecasting",
        "title": "Methods of Forecasting Grain Crop Yield Indicators Taking Into Account the Influence of Meteorological Conditions",
        "domain": "agriculture / climate",
        "chunks": [
            "Forecasting crop yields is one of the key challenges for the "
            "agricultural sector, especially in the context of a changing climate "
            "and unstable weather conditions. Kazakhstan, possessing significant "
            "territories suitable for growing grain crops, faces many challenges "
            "related to the effective management of agricultural activities. In "
            "this regard, yield forecasting becomes an integral part of planning "
            "and decision-making processes in agriculture. Information and "
            "analytical subsystems that integrate yield forecasting methods allow "
            "agribusinesses to estimate future production more accurately, "
            "minimise risks associated with climate change and optimise resource use."
        ],
    },
    {
        "source_id": "eval-fixture-asdae-collab-filtering",
        "title": "Auxiliary Stacked Denoising Autoencoder based Collaborative Filtering Recommendation",
        "domain": "recommender systems",
        "chunks": [
            "In recent years, deep learning techniques have achieved tremendous "
            "successes in natural language processing, speech recognition and "
            "image processing. Collaborative filtering (CF) recommendation is one "
            "of widely used methods and has significant effects in implementing "
            "the new recommendation function, but it also has limitations in "
            "dealing with the problem of poor scalability, cold start and data "
            "sparsity. Combining the traditional recommendation algorithm with "
            "the deep learning model has brought great opportunity for the "
            "construction of a new recommender system. We propose a novel "
            "collaborative recommendation model based on auxiliary stacked "
            "denoising autoencoder (ASDAE)."
        ],
    },
    {
        "source_id": "eval-fixture-cross-corpus-speech-emotion",
        "title": "Analysis of Deep Learning Architectures for Cross-Corpus Speech Emotion Recognition",
        "domain": "speech / audio ML",
        "chunks": [
            "Speech Emotion Recognition (SER) is an important and challenging "
            "task for human-computer interaction. In the literature deep "
            "learning architectures have been shown to yield state-of-the-art "
            "performance on this task when the model is trained and evaluated on "
            "the same corpus. However, prior work has indicated that such "
            "systems often yield poor performance on unseen data. To improve the "
            "generalisation capabilities of emotion recognition systems one "
            "possible approach is cross-corpus training, which consists of "
            "training the model on an aggregation of different corpora."
        ],
    },
    {
        "source_id": "eval-fixture-brain-cancer-multimodal-survival",
        "title": "Survival Prediction of Brain Cancer with Incomplete Radiology, Pathology, Genomics, and Demographic Data",
        "domain": "oncology / genomics",
        "chunks": [
            "Integrating cross-department multi-modal data (e.g., radiological, "
            "pathological, genomic, and clinical data) is ubiquitous in brain "
            "cancer diagnosis and survival prediction. To date, such an "
            "integration is typically conducted by human physicians (and panels "
            "of experts), which can be subjective and semi-quantitative. Recent "
            "advances in multi-modal deep learning, however, have opened a door "
            "to leverage such a process to a more objective and quantitative "
            "manner. Prior art using four modalities on brain cancer survival "
            "prediction is limited by a 'complete modalities' setting (i.e., with "
            "all modalities available)."
        ],
    },
    {
        "source_id": "eval-fixture-adp-fl-differential-privacy",
        "title": "An Asynchronous Federated Learning Aggregation Method Based on Adaptive Differential Privacy",
        "domain": "privacy / federated learning",
        "chunks": [
            "Federated learning is a distributed machine learning technique that "
            "allows multiple devices to collaborate on learning a shared model "
            "without exchanging data. It can be used to improve model accuracy "
            "while protecting user privacy. However, traditional federated "
            "learning is vulnerable to attacks from generative adversarial "
            "networks (GANs). As a new privacy protection method, differential "
            "privacy enhances privacy protection capabilities by sacrificing some "
            "data accuracy. To optimize the privacy budget allocation scheme in "
            "traditional differential privacy, we propose a differential privacy "
            "method called ADP-FL, which dynamically adjusts the privacy budget "
            "based on Newton's Law of Cooling."
        ],
    },
    {
        "source_id": "eval-fixture-frustration-free-hamiltonian",
        "title": "When a Local Hamiltonian Must Be Frustration-Free",
        "domain": "quantum computing",
        "chunks": [
            "Quantum computers promise computational power qualitatively "
            "superior to that achievable classically. This power will not be "
            "unlimited: beyond much-touted applications, such as breaking "
            "encryption schemes, entire classes of problems are known to be "
            "intractable even for quantum computers. This work addresses a "
            "question of great practical relevance: in between these two "
            "extremes of certain (in)tractability, how can one efficiently "
            "diagnose the nature and properties of a given problem instance? To "
            "achieve this, we adopt a strategy of transferring insights from "
            "statistical physics and classical computing into the quantum realm."
        ],
    },
]
