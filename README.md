# Personalized Nanoparticle Drug Delivery System

**Multi-Agent Reinforcement Learning and Genomic Analysis for Precision Cancer Nanotherapy**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project presents a computational framework that integrates biophysical simulation, multi-agent reinforcement learning, and Vision Transformer-based genomic analysis to design personalized nanoparticle drug delivery strategies for cancer therapy.

### Status and data provenance

> **Synthetic simulation.** This is a research prototype. All patient inputs are **synthetic** (randomly generated sequence arrays); no TCGA/GDC data is included or used, and no clinical claims are made. The genomic model has **no validated AUROC**: earlier figures (0.71 / 0.79) were not backed by a real cohort or saved evaluation outputs and have been removed.
>
> Simulation metrics (hit rate, selectivity, drug utilisation) are produced by `demo.py` and the `validation/` scripts on a simplified tumour model. They are not yet saved as raw artifacts with seeds, so no headline numbers are reported here.

## Results Visualization

### System Demo
![Demo Results](images/demo_results.png)

### Baseline Comparison
![Baseline Comparison](images/baseline_comparison.png)

### Ablation Studies
![Ablation Results](images/ablation_results.png)

### Metrics
![Metrics](images/metrics_visualization.png)

## Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATED PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Patient    │───▶│   Genomic    │───▶│    Drug      │      │
│  │   Genomic    │    │    ViT       │    │  Response    │      │
│  │   Profile    │    │   Model      │    │  Prediction  │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
│                                                  ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Biophysical │◀───│  Multi-Agent │◀───│   Reward     │      │
│  │  Simulation  │    │      RL      │    │ Modulation   │      │
│  │  (Brownian)  │    │  (5 Agents)  │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Personalized Treatment Plan              │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Biophysical Simulation**: Overdamped Langevin dynamics for realistic Brownian motion of nanoparticles
- **3-Zone Tumor Model**: Necrotic core, hypoxic zone, and proliferative rim with distinct properties
- **Multi-Agent RL**: 5 autonomous nanoparticle agents with hierarchical decision-making
- **Pheromone Communication**: Bio-inspired inter-agent coordination
- **Vision Transformer**: 6.4M parameter model for genomic drug response prediction (trained on synthetic inputs only)
- **Synthetic patients**: genomic inputs are randomly generated; a real-data path (TCGA/GDC acquisition, patient-level split) is future work

## Project Structure
```
amgen_project/
├── physics/
│   └── diffusion.py              # Brownian motion simulation
├── enviroment/
│   ├── tumor.py                  # 3-zone tumor microenvironment
│   └── barriers.py               # Endothelial barrier (EPR effect)
├── rl/
│   ├── environment.py            # RL environment
│   ├── hierarchical_agent.py     # High/low-level policies
│   ├── communication.py          # Pheromone signaling
│   └── personalized_env.py       # Genomic-aware environment
├── models/
│   └── genomic_vit.py            # Vision Transformer architecture
├── genomic/
│   ├── feature_engineering.py    # SNP & clinical features
│   └── sequence_encoder.py       # DNA one-hot encoding
├── data/
│   └── genomic/
│       ├── raw/                  # TCGA data (generated)
│       └── processed/            # Encoded sequences
├── analytics/
│   ├── ablation_studies.py       # Component importance
│   ├── baseline_comparison.py    # vs Random, Greedy, Flat RL
│   ├── sensitivity_analysis.py   # Parameter robustness
│   └── compute_metrics.py        # Comprehensive evaluation
├── demo.py                       # Full pipeline demonstration
├── train_personalized_agents.py  # RL training script
└── requirements.txt
```

## Installation
```bash
git clone https://github.com/iencislennon/Nanoparticle-Drug-Delivery-System-Using-Multi-Agent-Reinforcement-Learning.git
cd Nanoparticle-Drug-Delivery-System-Using-Multi-Agent-Reinforcement-Learning

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Quick Start
```bash
python demo.py
```

## License

MIT License

## Contact

**Ansar Tleubayev**  
Department of Information Systems  
Suleyman Demirel University, Almaty, Kazakhstan
