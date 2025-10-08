# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Human Motion Analysis project for biomechanical data analysis focusing on age-related gait patterns. The project includes data preprocessing, exploratory data analysis, Statistical Parametric Mapping (SPM), and an interactive Streamlit intelligence panel for biomechanics visualization.

## Development Setup

### Virtual Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter environment
jupyter notebook
# or
jupyter lab
```

### Running the Streamlit App
```bash
# Activate virtual environment first
source .venv/bin/activate

# Run the Streamlit intelligence panel
streamlit run app/app.py
```

### Key Dependencies
- **Data Science Core**: pandas>=2.0.0, numpy>=1.24.0, scipy>=1.10.0
- **Visualization**: matplotlib>=3.7.0, seaborn>=0.12.0, plotly>=5.15.0
- **Web Framework**: streamlit>=1.28.0 (for interactive intelligence panel)
- **ML & Stats**: scikit-learn>=1.3.0, Statistical Parametric Mapping libraries
- **Data Processing**: openpyxl>=3.1.0, xlrd>=2.0.0

## Project Architecture

### Core Components
1. **Data Pipeline**: Raw Excel data → preprocessing → analysis-ready CSV
2. **Analysis Notebooks**: Sequential workflow for biomechanical analysis
3. **Streamlit App**: Interactive intelligence panel (`app/app.py`)
4. **Results Output**: Automated figure and table generation

### Data Flow
```
data/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx
    ↓ (prep.ipynb)
data/processed/final_advanced_biomechanical_dataset.csv
    ↓ (analysis notebooks)
Reports, figures, and Streamlit visualizations
```

### Analysis Workflow
1. **prep.ipynb**: Data preprocessing and feature engineering
   - Processes raw biomechanical data from Nature Scientific Data publication
   - Creates 88 derived variables for comprehensive analysis
   - Outputs clean dataset to `data/processed/`

2. **eda.ipynb**: Main exploratory analysis
   - Comprehensive age-related gait biomechanics analysis
   - Statistical Parametric Mapping (SPM) integration
   - Four primary research questions addressing clinical applications

3. **app/app.py**: Interactive Streamlit intelligence panel
   - Real-time biomechanics visualization
   - Interactive data exploration interface
   - Clinical decision support tools

## Data Structure

### Input Data
- `data/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx`: Raw biomechanical data (138 adults, 21-86 years)
- `data/Metadatos_AbleBodiedAdults.xlsx`: Subject metadata and demographics
- `data/WalkingSpeed.xlsx`: Walking speed reference data

### Processed Data
- `data/processed/final_advanced_biomechanical_dataset.csv`: Main analysis dataset (88 variables)
- `data/processed/final_advanced_biomechanical_dictionary.csv`: Variable definitions and metadata
- `data/processed/MAT_normalizedData_AbleBodiedAdults_v06-03-23.csv`: Raw data in CSV format

## Statistical Methods & Analysis Framework

### Core Analysis Approaches
- **Traditional Statistics**: Correlation analysis, group comparisons, ANOVA
- **Advanced Regression**: Multivariate models with confounder control
- **Statistical Parametric Mapping (SPM)**: Continuous-time analysis across gait cycle
- **Froude Normalization**: Fr = v²/(L×g) for anthropometric control
- **Clinical Risk Assessment**: Evidence-based thresholds (walking speed <1.0 m/s)

### Research Framework
Four primary research questions:
1. Speed & efficiency effects (raw vs normalized)
2. Joint biomechanics across age groups (ankle, knee, hip)
3. Confounding variable control (anthropometric vs neuromuscular)
4. Clinical significance and fall risk indicators

## Development Guidelines

### Notebook Development
- Always run `prep.ipynb` first to ensure latest processed data
- Follow sequential workflow for comprehensive analysis
- Use virtual environment for all notebook execution
- Save outputs to appropriate directories (`img/`, results folders)

### Streamlit App Development
- Interactive panel located in `app/app.py`
- Integrates with processed datasets for real-time visualization
- Focus on clinical applications and biomechanical insights
- Use plotly for interactive visualizations

### Data Analysis Standards
- Control for walking speed in joint kinetic analyses
- Apply Froude normalization when separating anthropometric effects
- Include clinical thresholds in risk assessments
- Report effect sizes alongside statistical significance
- Use SPM for continuous gait cycle analysis (0-100%)

## File Organization

### Key Files
- `app/app.py`: Streamlit intelligence panel application
- `prep.ipynb`: Data preprocessing pipeline
- `eda.ipynb`: Main exploratory data analysis
- `requirements.txt`: Complete dependency specification

### Output Locations
- Processed data: `data/processed/`
- Analysis figures: `img/` directory
- Results and tables: `results/` directory
- Streamlit generates real-time visualizations

## Clinical Applications

This framework supports:
- **Functional Screening**: Walking speed as primary assessment tool
- **Targeted Interventions**: Joint-specific recommendations (especially ankle power)
- **Risk Stratification**: Multi-indicator fall risk assessment
- **Interactive Decision Support**: Streamlit panel for clinical use
- **Longitudinal Monitoring**: Framework for tracking biomechanical changes

## Common Development Tasks

### Data Processing
```bash
# Full pipeline from raw data
source .venv/bin/activate
jupyter notebook notebooks/prep.ipynb
# Run all cells to generate latest processed dataset
```

### Analysis Execution
```bash
# Run main analysis
jupyter notebook notebooks/eda.ipynb
# Execute sequentially for complete biomechanical analysis
```

### Interactive Panel
```bash
# Launch Streamlit intelligence panel
source .venv/bin/activate
streamlit run app/app.py
# Access at http://localhost:8501
```