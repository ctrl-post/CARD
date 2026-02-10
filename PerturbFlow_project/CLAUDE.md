# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PerturbFlow is a deep learning framework for predicting gene perturbation effects using graph neural networks and flow-based models. The project focuses on single-cell perturbation response prediction with applications in synthetic biology and drug discovery.

## Common Commands

### Training
```bash
# Train on specific dataset with default settings
python main.py --dataset norman --seed 1

# Train with custom configuration
python main.py --dataset adamson --config config/custom_config.json --seed 1
```

### Inference
```bash
# Run inference with trained model
python inference.py --model-path checkpoints/model_epoch_10.pt --dataset norman --output-prediction results/predictions.csv
```

### Model Evaluation
```bash
# Evaluate trained model with specific metrics
python evaluate_model.py --model-path checkpoints/best_model.pt --dataset dixit --metrics mse pearson
```

## High-Level Architecture

### Core Components

**Data Pipeline**:
- `PerturbationDataLoader` in `data/data_loader.py`: Main class for loading and preprocessing single-cell perturbation data
- Supports multiple datasets: norman, adamson, dixit, replogle_k562_essential, replogle_rpe1_essential
- Processes .h5ad files (AnnData format) containing single-cell gene expression data
- Creates PyTorch Geometric data objects for model training

**Models**:
- `PerturbFlowModel` in `models/perturbation_flow_model.py`: Main predictive model combining GNN and flow-based approaches
- `FlowModel` in `models/flow_model.py`: Flow-based generative components
- `InteractionModel` in `models/interaction_model.py`: Gene-gene interaction modeling
- `MLP` in `models/mlp.py`: Multi-layer perceptron components

**Training**:
- `PerturbFlowTrainer` in `training/trainer.py`: Centralized training orchestration
- `Evaluator` in `training/evaluator.py`: Model performance assessment

**Configuration**:
- Comprehensive configuration system in `config/config.py` with dataclass-based configuration classes
- Modular configs for data, model, training, evaluation, and project settings

### Data Flow Architecture
1. **Input**: Single-cell gene expression data in .h5ad format containing:
   - Observations: Individual cells with perturbation conditions
   - Variables: Gene expression measurements
   - Metadata: Cell annotations and perturbation labels

2. **Preprocessing**: 
   - Filters cells based on GPT embedding gene coverage
   - Creates matched perturbation gene sets
   - Generates PyTorch Geometric graph structures

3. **Model Pipeline**:
   - Input: Cell state (gene expression) + perturbation genes
   - Processing: GNN layers + Flow model transformations
   - Output: Predicted cell state after perturbation

4. **Evaluation**: 
   - MSE for expression prediction accuracy
   - Pearson correlation for biological relevance
   - Differential expression analysis validation

### Key Configuration Parameters
```python
{
  "data": {
    "split_type": "simulation",  # or "custom", "single", "no_test"
    "train_gene_set_size": 0.75,
    "seed": 1
  },
  "model": {
    "hidden_size": 64,
    "num_gene_gnn_layers": 1,
    "coexpression_threshold": 0.4
  },
  "training": {
    "epochs": 20,
    "learning_rate": 1e-3,
    "use_wandb": false
  }
}
```

### Supported Datasets
- norman: Single-cell perturbation dataset
- adamson: Alternative perturbation dataset
- dixit: Another perturbation dataset
- replogle_k562_essential: Essential genes perturbation in K562 cells
- replogle_rpe1_essential: Essential genes perturbation in RPE1 cells

### Model Components Integration
The architecture consists of:
1. **Gene Embedding Layer**: Processes GPT-based gene embeddings
2. **Graph Neural Network**: Models gene-gene interactions using co-expression networks
3. **Flow-based Model**: Learns transformation functions for perturbation effects
4. **Perturbation Interactions**: Models how genes interact under perturbation conditions
5. **Output Decoder**: Predicts final gene expression states after perturbation

All components are modular and can be configured independently through the configuration system.