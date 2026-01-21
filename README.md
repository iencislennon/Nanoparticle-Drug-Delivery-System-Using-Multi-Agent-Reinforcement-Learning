# Personalized Nanoparticle Drug Delivery System

**Multi-Agent Reinforcement Learning and Genomic Analysis for Precision Cancer Nanotherapy**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project presents a computational framework that integrates biophysical simulation, multi-agent reinforcement learning, and Vision Transformer-based genomic analysis to design personalized nanoparticle drug delivery strategies for cancer therapy.

### Key Results

| Metric | Baseline | Our System | Improvement |
|--------|----------|------------|-------------|
| Cancer Cell Hit Rate | 0.5% | 3.2% | **+540%** |
| Targeting Efficiency | — | — | **+52%** |
| Selectivity Index | 1.0× | 6.8× | **+580%** |
| Drug Utilization | 15% | 78% | **+420%** |
| Genomic AUROC | — | 0.79 | — |

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    INTEGRATED PIPELINE                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Patient    │───▶│   Genomic    │───▶│    Drug      │      │
│  │   Genomic    │    │    ViT       │    │  Response    │      │
│  │   Profile    │    │   Model      │    │  Prediction  │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │             │
│                                                  ▼             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Biophysical │◀───│  Multi-Agent │◀───│   Reward     │      │
│  │  Simulation  │    │      RL      │    │ Modulation   │      │
│  │  (Brownian)  │    │  (5 Agents)  │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                                      │
│         ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Personalized Treatment Plan              │     │
│  └──────────────────────────────────────────────────────┘      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Features

- **Biophysical Simulation**: Overdamped Langevin dynamics for realistic Brownian motion of nanoparticles
- **3-Zone Tumor Model**: Necrotic core, hypoxic zone, and proliferative rim with distinct properties
- **Multi-Agent RL**: 5 autonomous nanoparticle agents with hierarchical decision-making
- **Pheromone Communication**: Bio-inspired inter-agent coordination
- **Vision Transformer**: 6.4M parameter model for genomic drug response prediction
- **TCGA Integration**: Trained on 1,084 patients with 61,334 somatic mutations

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
# Clone repository
git clone https://github.com/iencislennon/personalized-nanoparticle-delivery.git
cd personalized-nanoparticle-delivery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Dataset (from TCGA)

```bash
python data/genomic/raw/download_minimal_dataset.py
```

### 2. Process Genomic Data

```bash
python genomic/feature_engineering.py
python genomic/sequence_encoder.py
```

### 3. Train Genomic Model

```bash
python data/genomic/processed/train_genomic_model.py
```

### 4. Run Full Demo

```bash
python demo.py
```

## Components

### Biophysical Simulation

The simulation uses the overdamped Langevin equation:

```
dx/dt = √(2D) · ξ(t)
```

Where diffusion coefficient D is calculated via Stokes-Einstein relation:

```
D = kᵦT / (6πηr)
```

For 100nm Fe₃O₄ nanoparticles in blood plasma at 310K: **D = 1.30 × 10⁻¹² m²/s**

### Tumor Microenvironment

| Zone | Radius | Oxygen (mmHg) | pH | Properties |
|------|--------|---------------|-----|------------|
| Necrotic | 0-50% | 0-5 | 6.2-6.5 | Dead cells, low density |
| Hypoxic | 50-80% | 5-20 | 6.5-6.8 | High IFP (15-25 mmHg) |
| Proliferative | 80-100% | 20-40 | 7.0-7.4 | EPR effect (200-400nm pores) |

### Reinforcement Learning

**State Space** (27 dimensions):
- Position & velocity (6D)
- Local tumor properties (6D)
- Oxygen & pressure gradients (6D)
- Neighboring agent positions (9D)

**Action Space** (8 actions):
- Move in 6 directions (±X, ±Y, ±Z)
- Release drug
- Wait

**Reward Structure**:
| Event | Reward |
|-------|--------|
| Cancer cell targeting | +100 |
| Necrotic core delivery | +50 |
| Cancer cell proximity | +10 |
| Healthy tissue damage | -5 |
| Time step penalty | -1 |
| Coordinated targeting | +20 |

### Genomic Vision Transformer

```
Input: DNA sequence (1024 bp, one-hot encoded)
  ↓
Patch Embedding (16 bp patches → 64 patches × 256 dim)
  ↓
Positional Encoding (sinusoidal)
  ↓
Transformer Encoder (8 layers, 8 heads)
  ↓
Classification Head → Drug Response + Uncertainty
```

**Model Statistics**:
- Parameters: 6,384,258
- Input: 1024 × 4 (one-hot DNA)
- Output: Drug response probability + uncertainty estimate

## Results

### Ablation Studies

| Component Removed | Performance Drop |
|-------------------|------------------|
| Strategic Planner | -35% |
| Inter-agent Communication | -28% |
| Genomic Personalization | -22% |

### Genomic Model Performance

- **AUROC**: 0.79 ± 0.03
- **Precision**: 0.74
- **Recall**: 0.71

Top predictive features:
1. TP53 mutation status
2. Overall mutation burden
3. CYP2D6 metabolizer phenotype
4. Tumor stage

## Outputs

Running the demo generates:

```
outputs/
├── demo_results.png              # Visualization
├── demo_report.txt               # Summary report
├── baseline_comparison.png       # vs baselines
├── ablation_results.png          # Component importance
├── sensitivity_analysis.png      # Parameter robustness
└── metrics_report.txt            # Detailed metrics
```

## Citation

```bibtex
@article{tleubayev2025nanoparticle,
  title={Personalized Nanoparticle Drug Delivery System Using Multi-Agent 
         Reinforcement Learning and Genomic Analysis},
  author={Tleubayev, Ansar},
  journal={arXiv preprint},
  year={2025},
  institution={Suleyman Demirel University}
}
```

## Acknowledgments

- Suleyman Demirel University for computational resources
- The Cancer Genome Atlas (TCGA) Research Network for genomic data
- Amgen Scholars Program

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

**Ansar Tleubayev**  
Department of Information Systems  
Suleyman Demirel University, Almaty, Kazakhstan

---

*This work sits at the intersection of computational biology, machine learning, and precision oncology, laying the groundwork for optimizing cancer nanomedicine entirely in silico.*