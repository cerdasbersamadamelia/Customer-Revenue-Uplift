# Customer Revenue Uplift Simulator

This project predicts and simulates customer revenue uplift for telecommunications marketing campaigns using machine learning. The complete workflow is implemented in `notebook/app.ipynb`.

## Overview

A machine learning system that uses T-learner methodology to predict which customers will generate the most revenue uplift from marketing campaigns. It integrates multiple data sources and provides simulation capabilities to optimize campaign ROI.

## Features

- **Multi-source data integration**: Customer profiles, usage metrics, campaign history, network KPIs, and complaints
- **Automated feature engineering**: Time-based aggregations, outlier detection, categorical encoding
- **T-learner uplift modeling**: Separate Random Forest models for treatment and control groups
- **Campaign ROI simulation**: Predict and compare different targeting strategies
- **Interactive visualizations**: Charts and plots using matplotlib, plotly, and gradio
- **Model explainability**: SHAP values for understanding feature importance
- **Network analysis**: Graph-based customer influence modeling

## Quick Start

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**: Place CSV files in the `dataset/` folder:

   - `customer_profile.csv` - Customer demographics and ARPU
   - `usage_metrics.csv` - Service usage patterns
   - `campaign_history.csv` - Historical campaign responses
   - `network_kpi.csv` - Network quality metrics
   - `complaints.csv` - Customer service data

3. **Run the notebook**:
   ```bash
   jupyter lab notebook/app.ipynb
   ```

## Workflow

The notebook follows this structured approach:

1. **Environment Setup**: Load libraries and configure parameters
2. **Data Loading**: Import and validate all CSV datasets
3. **Feature Engineering**: Create features with time windows and interactions
4. **Uplift Modeling**: Train T-learner models (treatment vs control)
5. **Evaluation**: Assess model performance and feature importance
6. **Simulation**: Run what-if scenarios for campaign optimization
7. **Visualization**: Generate interactive charts and explanations

## Key Components

- **ProjectConfig**: Centralized configuration class for all parameters
- **DataLoader**: Handles data import, validation, and cleaning
- **Feature Engineering**: Creates time-based and interaction features
- **T-learner Models**: Random Forest classifiers and regressors
- **SHAP Explainer**: Model interpretability and feature analysis
- **Interactive Dashboard**: Gradio-based interface for simulations

## Output

- Customer uplift scores and rankings
- Campaign ROI predictions and comparisons
- Feature importance analysis with SHAP values
- Interactive visualizations and simulations
- Performance metrics and model evaluation

## Configuration

Key parameters can be modified in the `ProjectConfig` class:

- Model parameters (Random Forest settings, cross-validation)
- Business logic (campaign costs, revenue thresholds)
- Feature engineering (time windows, network thresholds)
- Visualization settings (colors, figure sizes)

## Requirements

- Python 3.8+
- See `requirements.txt` for complete dependency list
- Jupyter Lab/Notebook for running the analysis

## Project Structure

See `PROJECT_STRUCTURE.md` for detailed folder organization.

## License

MIT License - see `LICENSE` file for details.

**Created by Damelia, 2025**
