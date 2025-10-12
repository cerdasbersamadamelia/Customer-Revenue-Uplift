# Customer Revenue Uplift Simulator - Project Overview

A machine learning system that predicts and simulates customer revenue uplift for telecommunications marketing campaigns using advanced T-learner methodology.

## What It Does

This system helps telecom companies optimize their marketing campaigns by:

- Identifying which customers are most likely to generate revenue uplift
- Predicting the incremental revenue impact of targeting specific customers
- Simulating different campaign strategies to maximize ROI
- Providing explainable insights into customer behavior patterns

## Technical Approach

**T-learner Methodology**: Uses separate machine learning models for treatment and control groups to estimate true incremental uplift, not just response probability.

**Multi-source Integration**: Combines customer demographics, usage patterns, network quality, campaign history, and service complaints for comprehensive customer understanding.

**Feature Engineering**: Automatically creates time-based aggregations, interaction features, and behavioral scores from raw data.

## Key Features

- **Data Integration**: Loads and validates 5 CSV datasets with automated quality checks
- **Feature Engineering**: Creates 50+ features including time windows (7, 30 days) and network interactions
- **Uplift Modeling**: Random Forest models for treatment/control with cross-validation
- **ROI Simulation**: Compares different targeting strategies and budget allocations
- **Explainability**: SHAP values show which features drive uplift predictions
- **Visualization**: Interactive charts using matplotlib, plotly, and gradio
- **Network Analysis**: Graph-based modeling of customer influence patterns

## Workflow Steps

1. **Environment Setup**: Configure libraries and parameters via ProjectConfig class
2. **Data Loading**: Import and validate customer, usage, campaign, network, and complaint data
3. **Data Cleaning**: Handle missing values, outliers, and data quality issues automatically
4. **Feature Engineering**: Create time-based features, network quality scores, and interactions
5. **Model Training**: Train separate Random Forest models for treatment and control groups
6. **Uplift Calculation**: Compute incremental impact predictions for each customer
7. **Evaluation**: Assess model performance using multiple metrics and SHAP analysis
8. **Simulation**: Run what-if scenarios for different campaign targeting strategies
9. **Dashboard**: Interactive interface for exploring results and running simulations

## Input Requirements

**Required CSV Files** (in `dataset/` folder):

- `customer_profile.csv`: Demographics, ARPU, plan details
- `usage_metrics.csv`: Voice, data, SMS usage patterns
- `campaign_history.csv`: Previous campaign responses and outcomes
- `network_kpi.csv`: Network quality metrics by location and time
- `complaints.csv`: Customer service interactions and resolutions

## Output Deliverables

- **Customer Scores**: Uplift probability and expected revenue for each customer
- **Campaign Recommendations**: Optimal targeting strategies and budget allocation
- **Feature Analysis**: SHAP importance scores showing key drivers of uplift
- **Performance Metrics**: Model accuracy, precision, recall, and business impact measures
- **Interactive Visualizations**: Charts for exploring patterns and running simulations
- **ROI Projections**: Expected return on investment for different campaign scenarios

## Technology Stack

- **Python**: Core programming language with Jupyter notebooks
- **pandas/numpy**: Data manipulation and numerical computing
- **scikit-learn**: Machine learning models and evaluation
- **matplotlib/plotly**: Data visualization and interactive charts
- **gradio**: Interactive dashboard interface
- **shap**: Model explainability and feature importance
- **networkx**: Graph analysis for customer influence modeling

## Configuration

All parameters are configurable through the `ProjectConfig` class:

- **Model Settings**: Random Forest parameters, cross-validation options
- **Business Logic**: Campaign costs (5000 IDR), revenue thresholds (100K IDR)
- **Feature Engineering**: Time windows, network quality thresholds
- **Visualization**: Color schemes, figure sizes, chart preferences

**Created by Damelia, 2025**
