# Project Structure

This document describes the organization of the Customer Revenue Uplift Simulator project.

## Directory Layout

```
Project - Customer Revenue Uplift/
│
├── requirements.txt         # Python package dependencies
├── config.toml              # Optional external configuration file
├── README.md                # Main project documentation
├── PROJECT_OVERVIEW.md      # Detailed project summary
├── PROJECT_STRUCTURE.md     # This file - project organization
├── LICENSE                  # MIT license terms
│
├── dataset/                 # Input data directory (CSV files)
│   ├── customer_profile.csv    # Customer demographics and ARPU data
│   ├── usage_metrics.csv       # Service usage patterns and behavior
│   ├── campaign_history.csv    # Historical campaign responses
│   ├── network_kpi.csv         # Network quality metrics by location
│   └── complaints.csv          # Customer service interactions
│
├── model/                   # Optional saved models and artifacts
│   └── (Generated during execution - models, plots, results)
│
└── notebook/                # Main analysis workspace
    ├── app.ipynb               # Complete workflow implementation
    └── notes.md                # Optional development notes
```

## File Descriptions

### Core Files

- **`app.ipynb`**: Main Jupyter notebook containing the entire workflow from data loading to simulation
- **`requirements.txt`**: All Python dependencies needed to run the project
- **`LICENSE`**: MIT license allowing free use and modification

### Data Files (Required)

All CSV files must be placed in the `dataset/` folder before running:

- **Customer Profile**: Demographics, plan type, ARPU, geographic data
- **Usage Metrics**: Voice, data, SMS usage with timestamps
- **Campaign History**: Previous marketing campaign results and customer responses
- **Network KPI**: Network quality measurements (RSRP, RSRQ, availability, etc.)
- **Complaints**: Customer service tickets and resolution data

### Configuration

- **`config.toml`**: Optional external configuration (parameters can also be set in ProjectConfig class)
- **ProjectConfig class**: Main configuration within the notebook for all parameters

### Output Location

- **`model/` folder**: Automatically created for saving trained models, visualizations, and results
- **Notebook cells**: Interactive output displayed within the Jupyter environment

## How It Works

1. **Single Entry Point**: All analysis runs through `notebook/app.ipynb`
2. **Data Integration**: Loads and validates all CSV files from `dataset/`
3. **Configuration**: Uses ProjectConfig class for all parameters and settings
4. **Output Generation**: Creates visualizations, models, and results within the notebook
5. **Optional Saving**: Can save models and artifacts to `model/` folder

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter and open the main notebook
jupyter lab notebook/app.ipynb

# Follow the notebook cells sequentially
```

## Dependencies

See `requirements.txt` for the complete list. Key libraries include:

- **pandas, numpy**: Data manipulation
- **scikit-learn**: Machine learning models
- **matplotlib, plotly**: Visualization
- **gradio**: Interactive dashboard
- **shap**: Model explainability
- **networkx**: Graph analysis

**Created by Damelia, 2025**
