# 🏃‍♀️ Human Motion Analysis: Age-Related Gait Biomechanics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://motionanalysisprojectdavidrodriguezcianca.streamlit.app/)
[![Dataset](https://img.shields.io/badge/Dataset-Nature%20Scientific%20Data-blue)](https://www.nature.com/articles/s41597-023-02767-y)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Overview

This project presents a comprehensive analysis of age-related changes in human gait biomechanics through an interactive **Biomechanics Intelligence Panel**. Using data from 138 healthy adults (21-86 years), we investigate how aging affects walking patterns, joint mechanics, and clinical fall risk indicators.

### 🎯 Key Research Questions
- How do spatiotemporal gait parameters change across age groups?
- Which joints show the most significant age-related ROM changes?
- How do joint moments and power adapt with aging?
- What are the clinical implications for fall risk assessment?

## 📊 Dataset Information

**Source:** [Nature Scientific Data (2024)](https://www.nature.com/articles/s41597-023-02767-y)  
**Citation:** Sivakanthan, S., Granata, K.P., Kesar, T.M. et al. An instrumented treadmill database for the study of healthy human locomotion over the full adult lifespan. *Sci Data* **11**, 22 (2024).  
**Sample:** 138 healthy adults (21-86 years)  
**Features:** 88+ biomechanical variables including:
- Spatiotemporal parameters (speed, cadence, step length/width)
- Joint kinematics (ROM, angles, variability)
- Joint kinetics (moments, efficiency)
- Joint power (generation, absorption)
- Muscle activity (EMG - 7 bilateral muscles)
- Ground reaction forces (GRF - 3D)

### 🔬 Study Methodology

**Experimental Setup:**
- **Setting:** Laboratory-controlled instrumented treadmill analysis
- **Protocol:** Self-selected walking speed on dual-belt treadmill system
- **Participants:** Rigorous screening for neurological/musculoskeletal conditions
- **Data Collection:** Multiple gait cycles recorded per participant

**Measurement Systems:**
- **3D Motion Capture:** Full-body kinematics with marker-based tracking
- **Force Plates:** Embedded dual-belt treadmill with 3D GRF measurement
- **EMG Recording:** 7 bilateral lower-limb muscles (gastrocnemius, rectus femoris, vastus lateralis, biceps femoris, semitendinosus, tibialis anterior, erector spinae)
- **Data Processing:** Time-normalized to 1001 points per gait cycle (0-100%)

**Data Quality:**
- Complete kinematic and kinetic data for all participants
- EMG data availability: ~77-79% (systematic technical limitations documented)
- No systematic missing data patterns across age groups
- Rigorous artifact detection and signal filtering protocols

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
