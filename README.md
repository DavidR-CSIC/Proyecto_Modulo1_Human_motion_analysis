# 🏃‍♀️ Human Motion Analysis: Age-Related Gait Biomechanics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://motionanalysisprojectdavidrodriguezcianca.streamlit.app/)
[![Dataset](https://img.shields.io/badge/Dataset-Nature%20Scientific%20Data-blue)](https://www.nature.com/articles/s41597-023-02767-y)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Overview

# Dataset Description

This project uses the full-body gait and motion capture dataset published by **Van Criekinge et al. (2023)** in *Scientific Data*, titled  
**“A full-body motion capture gait dataset of 138 able-bodied adults across the life span and 50 stroke survivors.”**  
📄 [Nature article link](https://www.nature.com/articles/s41597-023-02767-y)

---

## Overview

The dataset provides comprehensive **3D motion capture, ground reaction force, and electromyography (EMG)** data collected from:
- **138 able-bodied adults** (ages 21–86 years)
- **50 stroke survivors** (ages 19–85 years)

Each participant performed multiple walking trials under standardized conditions. Data were recorded using a **full-body Plug-in Gait marker model**, synchronized force plates, and surface EMG sensors.

The dataset is structured for both **raw data access** (C3D files) and **processed data analysis** (MATLAB `.mat` files).

---

## Contents

### 1. Participant Groups
| Group | Count | Age Range | Notes |
|-------|--------|------------|-------|
| Able-bodied | 138 | 21–86 years | Baseline gait data across lifespan |
| Stroke survivors | 50 | 19–85 years | Includes clinical metadata (stroke type, lesion location, time since stroke, functional scores) |

---

### 2. Raw Data (C3D Files)
Each C3D file contains:
- 3D marker trajectories (full-body Plug-in Gait model)
- Ground reaction forces (from dual force plates)
- Raw EMG signals (14 bilateral muscles)
- Anthropometrics and trial metadata

The C3D files are organized in two main directories:

---

### 3. Processed Data (MAT Files)
Preprocessed and stride-normalized MATLAB structures (`.mat`) include:
- **Kinematics:** joint angles, segment positions, and center of mass trajectories  
- **Kinetics:** joint moments, powers, and ground reaction forces (normalized to body mass)  
- **EMG:** rectified, filtered, and time-normalized signals for 14 muscles  
- **Gait events:** heel strike, toe-off, stride indices, and stride time normalization  
- **Metadata:** participant info, anthropometrics, and trial notes  

A detailed variable description is provided in the included  
`MATdatafiles_description_v1.3_LST` document.

---

### 4. Data Structure Summary

| Data Type | Variables | Notes |
|------------|------------|-------|
| **Kinematics** | 3D joint angles, segment positions, CoM trajectories | 1000 time-normalized points per stride |
| **Kinetics** | Joint moments, powers, GRFs | Force-plate-validated strides only |
| **EMG** | 14 bilateral muscles | Raw and processed versions |
| **Events & Metadata** | Gait event timings, stride IDs, anthropometrics | Event markers aligned with C3D frames |

> ⚠️ Some strides and trials may have missing EMG or force data. Missing values are represented as `NaN` in processed files.

---

### 5. Tools and Access

- The full dataset (raw and processed) is hosted on **Figshare**, linked through the Nature article.  
- Example MATLAB code for loading and visualizing the data is included with the dataset.  
- A **MatToPy** converter is available for importing MATLAB structures into Python workflows.  
- Compatible with **Visual3D**, **MATLAB**, and **Python c3d readers**.


---

## Citation

If you use this dataset in your work, please cite:

> Van Criekinge, T., De Kegel, A., Van Campenhout, A. *et al.*  
> **A full-body motion capture gait dataset of 138 able-bodied adults across the life span and 50 stroke survivors.**  
> *Scientific Data* 10, 767 (2023).  
> DOI: [10.1038/s41597-023-02767-y](https://doi.org/10.1038/s41597-023-02767-y)

---




## 🏗️ Project Structure

```
Proyecto_Modulo1_Human_motion_analysis/
├── 📱 app/
│   └── app.py                           # Main Streamlit application
├── 📊 data/
│   ├── processed/                       # Processed datasets
│   │   ├── final_advanced_biomechanical_dataset.csv
│   │   ├── final_advanced_biomechanical_dictionary.csv
│   │   └── MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx
│   ├── Metadatos_AbleBodiedAdults.xlsx  # Subject metadata
│   └── WalkingSpeed.xlsx                # Speed reference data
├── 📓 notebooks/
│   ├── eda.ipynb                        # Exploratory Data Analysis
│   ├── eda2.ipynb                       # Advanced EDA
│   ├── eda3.ipynb                       # Statistical Analysis
│   ├── prep.ipynb                       # Data Preprocessing
│   └── spm.ipynb                        # Statistical Parametric Mapping
├── 📈 reports/                          # Analysis reports and figures
├── 🎯 results/                          # Model outputs and findings
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## 🌐 Live Demo

**🚀 Try the app online:** [https://motionanalysisprojectdavidrodriguezcianca.streamlit.app/](https://motionanalysisprojectdavidrodriguezcianca.streamlit.app/)

*The interactive Biomechanics Intelligence Panel is deployed and ready to use - no installation required!*

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.8 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Proyecto_Modulo1_Human_motion_analysis.git
cd Proyecto_Modulo1_Human_motion_analysis
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`

## 🔍 Features

### 📊 Interactive Dashboard Tabs

1. **Dataset Overview**: Complete dataset characteristics and feature exploration
2. **Spatiotemporal Analysis**: Walking speed patterns and clinical risk assessment
3. **Joint ROM Analysis**: Range of motion changes across age groups
4. **Joint Moments**: Strength and efficiency analysis with clinical interpretation
5. **Joint Power**: Power generation/absorption patterns and age-related changes
6. **Gait Cycle Explorer**: Individual subject time-series visualization
7. **Subject Explorer**: Detailed biomechanical profiles with radar charts
8. **Help & About**: Methodology, clinical thresholds, and data provenance

### 🎛️ Interactive Controls
- Age group filtering (Young/Adult/Senior)
- Age range sliders
- Gender-based analysis
- Clinical threshold overlays
- Variable category selection
- Data export functionality

### 📈 Advanced Analytics
- **Statistical Analysis**: ANOVA, correlations, effect sizes
- **Clinical Thresholds**: Fall risk assessment (< 1.0 m/s, < 0.8 m/s)
- **Visualization**: Professional color schemes and interactive plots
- **Time-Series Analysis**: Complete gait cycle exploration
- **Individual Profiling**: Subject-specific biomechanical patterns

## 📋 Key Findings

### 🚶‍♂️ Spatiotemporal Changes
- Walking speed declines ~0.5-1% per year after age 60
- Increased gait variability with aging
- Clinical fall risk thresholds effectively identify at-risk individuals

### 🦵 Joint Mechanics
- **Ankle**: Most pronounced age-related power decline
- **Hip**: Compensatory strategy increases to maintain walking speed
- **Knee**: Variable preservation patterns across individuals

### ⚠️ Clinical Implications
- Speed < 1.0 m/s: Increased fall risk
- Speed < 0.8 m/s: High fall risk, mobility impairment
- Joint power hierarchy shifts with aging

## 🔬 Methodology

### Data Processing
- Quality-controlled preprocessing pipeline
- Body mass normalization for kinetics
- Froude number normalization for speed
- Z-score standardization for comparisons

### Statistical Approaches
- **ANOVA** for group comparisons
- **Pearson/Spearman** correlations
- **Effect size calculations** (η²)
- **Clinical threshold analysis**
- **Statistical Parametric Mapping** (SPM1D) for time-series

### Visualization
- Professional color palette optimized for accessibility
- Interactive Plotly visualizations
- Responsive design for various screen sizes
- Export capabilities for figures and data

## 🛠️ Technical Details

### Core Dependencies
```python
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
streamlit>=1.28.0      # Web application framework
plotly>=5.15.0         # Interactive visualizations
scipy>=1.10.0          # Statistical analysis
scikit-learn>=1.3.0    # Machine learning utilities
matplotlib>=3.7.0      # Static plotting
seaborn>=0.12.0        # Statistical visualization
```

### Performance Optimizations
- **Caching**: `@st.cache_data` for expensive computations
- **Lazy loading**: Data loaded on demand
- **Memory management**: Efficient data structures
- **Background processing**: Parallel Streamlit instances

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -am 'Add new analysis'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Create a Pull Request

## 📚 References

1. **Primary Dataset**: Camargo-Junior et al. (2024). "A comprehensive dataset of healthy adult gait biomechanics." *Nature Scientific Data*. [DOI: 10.1038/s41597-023-02767-y](https://www.nature.com/articles/s41597-023-02767-y)

2. **Clinical Thresholds**:
   - Studenski et al. (2011). "Gait speed and survival in older adults." *JAMA*.
   - Cesari et al. (2005). "Prognostic value of usual gait speed in well-functioning older people." *JAGS*.

3. **Statistical Methods**:
   - Pataky (2010). "Generalized n-dimensional biomechanical field analysis using statistical parametric mapping." *Journal of Biomechanics*.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**David Rodríguez Cianca**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/david-rodriguez-cianca/)

*Data Science & AI Bootcamp Project - Biomechanics Intelligence Panel*

---

## 🔄 Version History

- **v1.0.0** (2024): Initial release with interactive dashboard
- **v1.1.0** (2024): Added gait cycle explorer and subject profiling
- **v1.2.0** (2024): Enhanced color palette and dataset attribution

## 📞 Support

For questions, issues, or contributions:
1. Check the [Issues](https://github.com/your-username/Proyecto_Modulo1_Human_motion_analysis/issues) page
2. Open a new issue with detailed description
3. Contact via LinkedIn for collaboration opportunities

---

*This project demonstrates the application of data science techniques to biomechanical research, providing insights into age-related changes in human movement patterns with clinical relevance for fall risk assessment and mobility health.*
