# In-Context Learning Strategies Emerge Rationally

This repository contains the code for "In-Context Learning Strategies Emerge Rationally", which presents a normative Bayesian framework explaining why transformers learn different in-context learning (ICL) strategies. The research demonstrates that ICL strategies can be understood as optimal adaptations to data given computational constraints, using rational analysis to predict transformer behavior without access to model weights.

## Overview

This work addresses a fundamental question: *why* do transformers learn different ICL strategies under varying conditions? Using a hierarchical Bayesian framework grounded in rational analysis, we show that ICL behavior emerges from an optimal tradeoff between strategy loss and complexity.

### Key Contributions

- **Normative Framework**: A Bayesian framework that predicts transformer next-token predictions throughout training without access to model weights
- **Strategy Emergence**: Explains why models transition from generalizing to memorizing strategies as task diversity increases
- **Rational Analysis**: Views pretraining as updating posterior probabilities over strategies, with inference as posterior-weighted averaging
- **Empirical Validation**: Demonstrates framework predictions match transformer behavior across multiple ICL tasks

### ICL Tasks Studied

The codebase supports three different in-context learning tasks:
- **Balls & Urns (Categorical Sequence Prediction)**: Learning to predict distributions in categorical sequences by drawing from urns. This task captures the belief update interpretation of ICL.
- **Linear Regression**: Performing regression from input-output examples in context
- **Binary Classification**: Classifying examples based on contextual patterns

### Bayesian Predictors

Our framework identifies two primary ICL strategies:
- **Memorizing Predictor**: Assumes discrete prior over seen training tasks
- **Generalizing Predictor**: Uses continuous prior matching the true task distribution

## Setup

### Environment Configuration
Create a `.env` file in the project root with:
```bash
CACHE_DIR=/path/to/your/cache/directory
```
This directory will store all generated data, model checkpoints, and results.

### Dependencies
The project uses standard ML libraries including:
- PyTorch and Transformers (HuggingFace)
- NumPy, pandas for data handling
- Matplotlib, seaborn for visualization
- YAML for configuration management

## Quick Start

### 1. Generate Data
Generate training and evaluation datasets for all tasks:
```bash
src/scripts/generate_data.sh
```

Or generate data for a specific task:
```bash
python3 -m src.code.generate_data --setting categorical-sequence --num_dims 8 12 16 --random_seed 1
```

### 2. Create Experiment Configuration
Generate YAML configuration files for training runs:
```bash
python -m src.config my-experiment-name --setting linear-regression --num_dims_lst 8 --context_length_lst 16
```

This creates experiment configs in `../experiments/<setting>/my-experiment-name/yaml-configs/`

### 3. Train Models
Train on all configurations in an experiment:
```bash
python -m src.train ../experiments/<setting>/my-experiment-name
```

Or train a single configuration:
```bash
python -m src.train ../experiments/<setting>/my-experiment-name/yaml-configs/<config-file>.yaml
```

### 4. Run Analysis
Analyze trained models and generate figures:
```bash
python -m src.code.analysis_pipeline --task categorical-sequence
```

For interactive analysis, use the Jupyter notebooks in `src/notebooks/`.

## Project Structure

```
src/
├── code/                     # Main source code
│   ├── config.py            # Configuration and experiment setup
│   ├── generate_data.py     # Data generation for different tasks
│   ├── train.py             # Training loop with HuggingFace Transformers
│   ├── models.py            # Transformer model definitions
│   ├── analysis_pipeline.py # Automated analysis pipeline
│   ├── analysis_utils.py    # Analysis utilities and metrics
│   ├── utils.py             # Dataset utilities and callbacks
│   └── make_figs.py         # Figure generation
├── notebooks/               # Jupyter notebooks for interactive analysis
├── scripts/                 # Shell scripts for batch operations
experiments/                 # Generated experiment configurations
figures/                     # Generated analysis figures
results/                     # Analysis results and correlations
```

## Configuration Options

Key parameters for experiments:
- `setting`: Task type (categorical-sequence, linear-regression, classification)
- `num_dims`: Dimensionality of the task (8, 12, 16)
- `context_length`: Number of examples in context (16, 32, 64, 128, 256, 320)
- `mlp_expansion_factor`: MLP width multiplier (0.5, 1, 2, 4, 8, 12, 16, 24, 32)
- `num_tasks`: Number of training tasks (powers of 2 from 4 to 4096)
- `max_steps`: Training steps (default: 100,000)

## Analysis Features

The analysis pipeline provides:
- **Bayesian Model Selection**: Decomposition of transformer behavior into memorizing vs. generalizing predictors
- **Posterior Probability Tracking**: Evolution of strategy preferences throughout training
- **Evidence Accumulation**: How models integrate information across context under different strategies
- **Task Diversity Effects**: Impact of training task variety on strategy transitions
- **Complexity-Loss Tradeoffs**: Analysis of how computational constraints shape strategy emergence
- **Transience Analysis**: Temporal dynamics showing transitions from generalization to memorization
- **Predictive Validation**: Comparing Bayesian framework predictions with actual transformer behavior

## Results

Generated figures and analysis results are organized by task in:
- `figures/<task>/` - Visualization outputs
- `results/` - Quantitative analysis results

## Citation

If you use this code, please cite:
```bibtex
@article{wurgaft2025rational,
  title={In-Context Learning Strategies Emerge Rationally},
  author={Daniel Wurgaft and Ekdeep Singh Lubana and Core Francisco Park and Hidenori Tanaka and Gautam Reddy and Noah D. Goodman},
  journal={arXiv preprint arXiv:2506.17859},
  year={2025}
}
```