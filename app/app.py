# -*- coding: utf-8 -*-
"""
Biomechanics Intelligence Panel
Human Motion Analysis Dashboard - Professional Intelligence Interface

Created for comprehensive analysis of age-related biomechanical changes in gait patterns.
Data source: Nature Scientific Data publication (138 healthy adults, 21-86 years)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent font-related issues
from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, f_oneway
import json
import base64
from io import BytesIO
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# PROFESSIONAL COLOR PALETTE
# ============================================================================
PROFESSIONAL_COLORS = {
    # Age Categories - Sophisticated and distinguishable
    'age_categories': {
        'Young': "#0C9B88",      # Professional blue
        'Adult': "#096246",      # Sophisticated purple
        'Senior': "#022D1F"      # Elegant red
    },

    # Gender Categories - Balanced and inclusive
    'gender': {
        'F': "#E6649E",          # Professional pink
        'M': "#7692F0"           # Professional blue
    },

    # Chart Elements
    'primary': "#7AA9EA",        # Main brand blue
    'secondary': "#A77CEF",      # Accent purple
    'success': "#3E9277",        # Professional green
    'warning': '#D97706',        # Warm orange
    'danger': '#DC2626',         # Alert red
    'info': '#0891B2',           # Information teal

    # Performance Rankings
    'rankings': ["#EFB748", "#AC9AD5", "#4F827B"],  # Gold, Purple, Red

    # Clinical thresholds
    'clinical': {
        'normal': '#059669',     # Green
        'risk': '#D97706',       # Orange
        'high_risk': '#DC2626'   # Red
    }
}

# Configure Streamlit page
st.set_page_config(
    page_title="Biomechanics Intelligence Panel",
    page_icon="🏃‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {PROFESSIONAL_COLORS['primary']};
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    .sub-header {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {PROFESSIONAL_COLORS['secondary']};
        margin: 1rem 0;
    }}
    .metric-card {{
        background: linear-gradient(90deg, {PROFESSIONAL_COLORS['primary']} 0%, {PROFESSIONAL_COLORS['secondary']} 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }}
    .kpi-value {{
        font-size: 2rem;
        font-weight: bold;
        color: white;
    }}
    .kpi-label {{
        font-size: 0.9rem;
        color: #f8f9fa;
        margin-top: 0.25rem;
    }}
    .clinical-threshold {{
        background: {PROFESSIONAL_COLORS['danger']};
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }}
    .normal-threshold {{
        background: {PROFESSIONAL_COLORS['success']};
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }}
    .significance-badge {{
        background: {PROFESSIONAL_COLORS['warning']};
        color: white;
        padding: 0.2rem 0.4rem;
        border-radius: 0.2rem;
        font-size: 0.8rem;
        font-weight: bold;
    }}
    .sidebar .stSelectbox label {{
        font-weight: 600;
        color: #2c3e50;
    }}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size: 1.1rem;
        font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)

# Data loading with caching
@st.cache_data
def load_all_datasets():
    """Load and cache all biomechanical datasets"""
    try:
        # Load processed datasets with encoding specification
        processed_data = pd.read_csv('data/processed/final_advanced_biomechanical_dataset.csv', encoding='utf-8')
        dictionary = pd.read_csv('data/processed/final_advanced_biomechanical_dictionary.csv', encoding='utf-8')

        # Load raw time-series data (Excel)
        excel_path = 'data/processed/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx'
        if os.path.exists(excel_path):
            # Load main data sheet
            raw_data = pd.read_excel(excel_path, sheet_name=0, engine='openpyxl')
        else:
            # Fallback to CSV version
            raw_data = pd.read_csv('data/processed/MAT_normalizedData_AbleBodiedAdults_v06-03-23.csv',
                                 encoding='utf-8', low_memory=False)

        # Load metadata
        metadata_path = 'data/Metadatos_AbleBodiedAdults.xlsx'
        if os.path.exists(metadata_path):
            metadata = pd.read_excel(metadata_path, engine='openpyxl')
        else:
            metadata = None

        # Load walking speed reference
        speed_path = 'data/WalkingSpeed.xlsx'
        if os.path.exists(speed_path):
            speed_reference = pd.read_excel(speed_path, engine='openpyxl')
        else:
            speed_reference = None

        return processed_data, dictionary, raw_data, metadata, speed_reference

    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
        st.error("Please ensure all data files are present in the correct directories:")
        st.code("""
        data/processed/final_advanced_biomechanical_dataset.csv
        data/processed/final_advanced_biomechanical_dictionary.csv
        data/processed/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx
        data/Metadatos_AbleBodiedAdults.xlsx
        data/WalkingSpeed.xlsx
        """)
        st.stop()

@st.cache_data
def calculate_statistics(data, group_col, value_col):
    """Calculate comprehensive statistics with caching"""
    if data.empty or group_col not in data.columns or value_col not in data.columns:
        return {}, np.nan, np.nan

    stats_dict = {}
    groups = data[group_col].dropna().unique()

    for group in groups:
        group_data = data[data[group_col] == group][value_col].dropna()
        if len(group_data) > 0:
            stats_dict[group] = {
                'mean': group_data.mean(),
                'std': group_data.std(),
                'median': group_data.median(),
                'q25': group_data.quantile(0.25),
                'q75': group_data.quantile(0.75),
                'count': len(group_data)
            }

    # ANOVA
    try:
        group_arrays = [data[data[group_col] == group][value_col].dropna().values
                       for group in groups if len(data[data[group_col] == group][value_col].dropna()) > 0]
        if len(group_arrays) > 1 and all(len(arr) > 0 for arr in group_arrays):
            f_stat, p_value = f_oneway(*group_arrays)
        else:
            f_stat, p_value = np.nan, np.nan
    except:
        f_stat, p_value = np.nan, np.nan

    return stats_dict, f_stat, p_value

@st.cache_data
def calculate_effect_size(data, group_col, value_col):
    """Calculate effect sizes (eta-squared) with caching"""
    if data.empty or group_col not in data.columns or value_col not in data.columns:
        return np.nan

    try:
        groups = data[group_col].dropna().unique()
        if len(groups) < 2:
            return np.nan

        group_arrays = [data[data[group_col] == group][value_col].dropna().values
                       for group in groups if len(data[data[group_col] == group][value_col].dropna()) > 0]

        if len(group_arrays) < 2:
            return np.nan

        # Calculate eta-squared
        total_mean = data[value_col].mean()
        ss_between = sum(len(group) * (np.mean(group) - total_mean)**2 for group in group_arrays)
        ss_total = sum((data[value_col] - total_mean)**2)

        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        return eta_squared
    except:
        return np.nan

def create_kpi_card(value, label, color="primary"):
    """Create a styled KPI card"""
    color_mapping = {
        "primary": PROFESSIONAL_COLORS['primary'],
        "success": PROFESSIONAL_COLORS['success'],
        "warning": PROFESSIONAL_COLORS['warning'],
        "danger": PROFESSIONAL_COLORS['danger'],
        "info": PROFESSIONAL_COLORS['info']
    }

    color_code = color_mapping.get(color, PROFESSIONAL_COLORS['primary'])

    st.markdown(f"""
    <div style="background: {color_code}; padding: 1rem; border-radius: 0.5rem; text-align: center; margin: 0.5rem 0;">
        <div style="font-size: 2rem; font-weight: bold; color: white;">{value}</div>
        <div style="font-size: 0.9rem; color: #f8f9fa; margin-top: 0.25rem;">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def extract_time_series_variables(raw_data):
    """Extract time-series variables from MAT Excel file for SPM analysis"""
    # Load the time-series gait variables from MAT Excel file
    mat_file = 'data/processed/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx'

    try:
        # Get a sample sheet to see available variables
        sample_df = pd.read_excel(mat_file, sheet_name='Sub01')
        time_series_vars = sample_df.columns.tolist()
        return time_series_vars
    except:
        return []

def prepare_spm_data(raw_data, selected_variable):
    """Prepare gait cycle time-series data for SPM analysis from MAT Excel file"""
    mat_file = 'data/processed/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx'
    main_df = pd.read_csv('data/processed/final_advanced_biomechanical_dataset.csv')

    try:
        excel_file = pd.ExcelFile(mat_file)
        subjects = [sheet for sheet in excel_file.sheet_names if sheet.startswith('Sub')]

        # Initialize containers for age groups
        young_data = []
        adult_data = []
        senior_data = []

        # Load data for age group comparison (first 30 subjects for performance)
        for subject_sheet in subjects[:30]:
            subject_num = int(subject_sheet[3:])
            main_subject_data = main_df[main_df['ID'] == f'SUBJ{subject_num}']

            if not main_subject_data.empty:
                ts_df = pd.read_excel(mat_file, sheet_name=subject_sheet)
                if selected_variable in ts_df.columns:
                    subject_info = main_subject_data.iloc[0]
                    age_group = subject_info['AgeCategory']
                    variable_data = ts_df[selected_variable].values

                    if age_group == 'Young':
                        young_data.append(variable_data)
                    elif age_group == 'Adult':
                        adult_data.append(variable_data)
                    elif age_group == 'Senior':
                        senior_data.append(variable_data)

        # Convert to numpy arrays
        gait_cycle_percent = np.linspace(0, 100, 1001)  # 0-100% gait cycle

        spm_data = {
            'Young': np.array(young_data) if young_data else np.array([]),
            'Adult': np.array(adult_data) if adult_data else np.array([]),
            'Senior': np.array(senior_data) if senior_data else np.array([]),
            'time_points': gait_cycle_percent
        }

        return spm_data, gait_cycle_percent

    except Exception as e:
        st.error(f"Error loading time-series data: {e}")
        return None, None

def main():
    # Load all datasets
    processed_data, dictionary, raw_data, metadata, speed_reference = load_all_datasets()

    # Main title with blue background
    st.markdown('''
    <div style="background: linear-gradient(90deg, #2563EB 0%, #3B82F6 100%);
                padding: 2rem; border-radius: 1rem; text-align: center; margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="color: white; font-size: 2.5rem; font-weight: 700; margin: 0;">
            🏃‍♀️ Biomechanics Intelligence Panel
        </h1>
        <p style="color: #E6F3FF; font-size: 1.2rem; margin: 0.5rem 0;">
            Human Motion Analysis Dashboard - Age-Related Gait Biomechanics
        </p>
        <p style="color: #B3D9FF; font-size: 1rem; margin: 0;">
            Dataset from: <a href="https://www.nature.com/articles/s41597-023-02767-y" target="_blank"
            style="color: #FFF; text-decoration: underline;">Nature Scientific Data (2024)</a> |
            138 healthy adults (21-86 years) | Laboratory gait analysis
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # ============================================================================
    # SIDEBAR CONTROLS
    # ============================================================================
    with st.sidebar:
        st.markdown("## 🎛️ Global Controls")

        # Age group filter
        age_categories = ['All'] + sorted(processed_data['AgeCategory'].unique().tolist())
        selected_age_category = st.selectbox("Age Group", age_categories)

        # Age range slider
        age_min, age_max = int(processed_data['Age'].min()), int(processed_data['Age'].max())
        age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))

        # Sex filter
        sex_options = ['All'] + sorted(processed_data['Sex'].unique().tolist())
        selected_sex = st.selectbox("Sex", sex_options)

        # Speed threshold toggles
        st.markdown("### Speed Thresholds")
        show_clinical_1_0 = st.checkbox("Highlight < 1.0 m/s (Clinical Risk)", value=True)
        show_clinical_0_8 = st.checkbox("Highlight < 0.8 m/s (High Risk)", value=False)

        # Variable category selector (exclude Classification)
        if not dictionary.empty:
            all_categories = sorted(dictionary['Category'].dropna().unique().tolist())
            # Remove Classification category
            categories = [cat for cat in all_categories if cat != 'Classification']
            selected_categories = st.multiselect("Variable Categories", categories, default=categories[:3] if len(categories) >= 3 else categories)
        else:
            selected_categories = []

        st.markdown("---")

        # Data export section
        st.markdown("### 📊 Export Options")
        if st.button("📥 Download Filtered Data"):
            # Create filtered data for download
            filtered_data_temp = processed_data.copy()

            if selected_age_category != 'All':
                filtered_data_temp = filtered_data_temp[filtered_data_temp['AgeCategory'] == selected_age_category]

            filtered_data_temp = filtered_data_temp[
                (filtered_data_temp['Age'] >= age_range[0]) &
                (filtered_data_temp['Age'] <= age_range[1])
            ]

            if selected_sex != 'All':
                filtered_data_temp = filtered_data_temp[filtered_data_temp['Sex'] == selected_sex]

            # Convert filtered data to CSV
            csv = filtered_data_temp.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="filtered_biomechanical_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

        if st.button("📊 Generate Summary Report"):
            st.info("Generating comprehensive analysis report...")

    # ============================================================================
    # APPLY FILTERS
    # ============================================================================
    filtered_data = processed_data.copy()

    if selected_age_category != 'All':
        filtered_data = filtered_data[filtered_data['AgeCategory'] == selected_age_category]

    filtered_data = filtered_data[
        (filtered_data['Age'] >= age_range[0]) &
        (filtered_data['Age'] <= age_range[1])
    ]

    if selected_sex != 'All':
        filtered_data = filtered_data[filtered_data['Sex'] == selected_sex]

    # ============================================================================
    # GLOBAL KPI CARDS
    # ============================================================================
    st.markdown("## 📊 Dataset Overview")

    if not filtered_data.empty:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            create_kpi_card(len(filtered_data), "Subjects", "primary")

        with col2:
            create_kpi_card(len(processed_data.columns), "Features", "success")

        with col3:
            age_range_text = f"{filtered_data['Age'].min():.0f}-{filtered_data['Age'].max():.0f}"
            create_kpi_card(age_range_text, "Age Range", "warning")

        with col4:
            mean_age = f"{filtered_data['Age'].mean():.1f}y"
            create_kpi_card(mean_age, "Mean Age", "primary")

        with col5:
            sex_balance = f"{(filtered_data['Sex'] == 'F').sum():.0f}F / {(filtered_data['Sex'] == 'M').sum():.0f}M"
            create_kpi_card(sex_balance, "Sex Balance", "success")
    else:
        st.warning("No data available with current filters. Please adjust your selection.")
        return

    # ============================================================================
    # MAIN CONTENT TABS
    # ============================================================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Dataset Overview",
        "🚶‍♂️ Spatiotemporal",
        "🦵 Joint ROM",
        "💪 Joint Moments",
        "⚡ Joint Power",
        "🔍 Gait Cycle Explorer",
        "👤 Subject Explorer",
        "ℹ️ Help & About"
    ])

    # ============================================================================
    # TAB 1: DATASET OVERVIEW
    # ============================================================================
    with tab1:
        st.markdown("## 📊 Dataset Characteristics")

        col1, col2 = st.columns(2)

        with col1:
            # Age distribution
            fig_age = px.histogram(
                filtered_data,
                x='Age',
                color='AgeCategory',
                title="Age Distribution by Category",
                nbins=20,
                color_discrete_map=PROFESSIONAL_COLORS['age_categories']
            )
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)

            # BMI distribution by sex
            if 'BMI' in filtered_data.columns:
                fig_bmi = px.box(
                    filtered_data,
                    x='Sex',
                    y='BMI',
                    color='AgeCategory',
                    title="BMI Distribution by Sex and Age Category",
                    color_discrete_map=PROFESSIONAL_COLORS['age_categories']
                )
                fig_bmi.update_layout(height=400)
                st.plotly_chart(fig_bmi, use_container_width=True)

        with col2:
            # Feature category pie chart (all features, exclude Classification)
            if not dictionary.empty:
                # Get all categories except Classification for complete overview
                all_category_counts = dictionary[dictionary['Category'] != 'Classification']['Category'].value_counts()
                fig_pie = px.pie(
                    values=all_category_counts.values,
                    names=all_category_counts.index,
                    title="Complete Feature Distribution by Category"
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

            # Sex by age group stacked bar
            age_sex_counts = filtered_data.groupby(['AgeCategory', 'Sex']).size().reset_index(name='Count')
            fig_stack = px.bar(
                age_sex_counts,
                x='AgeCategory',
                y='Count',
                color='Sex',
                title="Subject Distribution: Sex by Age Category",
                color_discrete_map=PROFESSIONAL_COLORS['gender']
            )
            fig_stack.update_layout(height=400)
            st.plotly_chart(fig_stack, use_container_width=True)

        # Feature exploration by category
        st.markdown("### 🔍 Available Features by Category")

        # Define feature categories based on data dictionary
        feature_categories = {
            'Demographics & Anthropometrics': {
                'description': 'Basic subject characteristics and body measurements',
                'features': ['Age', 'Sex', 'BodyMass_kg', 'Height_mm', 'Height_m', 'LegLength_mm', 'LegLength_m', 'BMI', 'LegToHeight_Ratio', 'BMI_Category']
            },
            'Locomotion & Speed': {
                'description': 'Walking speed characteristics and asymmetry metrics',
                'features': ['Lside_mps', 'Rside_mps', 'AvgSpeed_mps', 'SpeedAsymmetry_abs', 'SpeedAsymmetry_pct', 'NormalizedSpeed', 'FroudeNumber', 'SpeedCategory']
            },
            'Joint Kinematics (ROM & Angles)': {
                'description': 'Joint range of motion, mean angles, and variability measurements',
                'features': ['Ankle_ROM', 'Ankle_Mean_Angle', 'Ankle_Angle_Variability', 'Knee_ROM', 'Knee_Mean_Angle', 'Knee_Angle_Variability', 'Hip_ROM', 'Hip_Mean_Angle', 'Hip_Angle_Variability', 'Pelvis_ROM', 'Pelvis_Mean_Angle', 'Pelvis_Angle_Variability']
            },
            'Joint Kinetics (Moments)': {
                'description': 'Joint moment measurements and efficiency metrics',
                'features': ['Ankle_Avg_Moment', 'Ankle_Peak_Moment', 'Ankle_Moment_Variability', 'Knee_Avg_Moment', 'Knee_Peak_Moment', 'Knee_Moment_Variability', 'Hip_Avg_Moment', 'Hip_Peak_Moment', 'Hip_Moment_Variability', 'Ankle_Moment_Efficiency', 'Knee_Moment_Efficiency', 'Hip_Moment_Efficiency']
            },
            'Joint Power': {
                'description': 'Joint power generation and absorption measurements',
                'features': ['Ankle_Avg_Power', 'Ankle_Peak_Power', 'Ankle_Power_Variability', 'Knee_Avg_Power', 'Knee_Peak_Power', 'Knee_Power_Variability', 'Hip_Avg_Power', 'Hip_Peak_Power', 'Hip_Power_Variability']
            },
            'Muscle Activity (EMG)': {
                'description': 'Electromyographic activity measurements for major muscle groups',
                'features': ['GAS_iEMG', 'GAS_Avg_EMG', 'GAS_Peak_EMG', 'RF_iEMG', 'RF_Avg_EMG', 'RF_Peak_EMG', 'VL_iEMG', 'VL_Avg_EMG', 'VL_Peak_EMG', 'BF_iEMG', 'BF_Avg_EMG', 'BF_Peak_EMG', 'ST_iEMG', 'ST_Avg_EMG', 'ST_Peak_EMG', 'TA_iEMG', 'TA_Avg_EMG', 'TA_Peak_EMG', 'ERS_iEMG', 'ERS_Avg_EMG', 'ERS_Peak_EMG']
            },
            'Ground Reaction Forces': {
                'description': 'Ground reaction force measurements in three directions',
                'features': ['GRF_Anteroposterior_Peak', 'GRF_Anteroposterior_Avg', 'GRF_Anteroposterior_Variability', 'GRF_Mediolateral_Peak', 'GRF_Mediolateral_Avg', 'GRF_Mediolateral_Variability', 'GRF_Vertical_Peak', 'GRF_Vertical_Avg', 'GRF_Vertical_Variability']
            },
            'Derived Metrics & Classifications': {
                'description': 'Computed indices and categorical classifications',
                'features': ['AgeCategory', 'AsymmetryCategory', 'GaitEfficiency', 'Speed_ZScore', 'BMI_ZScore', 'PerformanceScore']
            }
        }

        # Create expandable sections for each category
        for category, info in feature_categories.items():
            with st.expander(f"📊 **{category}** ({len(info['features'])} features)", expanded=False):
                st.markdown(f"*{info['description']}*")

                # Check which features are available in the dataset
                available_features = [f for f in info['features'] if f in processed_data.columns]
                missing_features = [f for f in info['features'] if f not in processed_data.columns]

                if available_features:
                    st.markdown(f"**✅ Available Features ({len(available_features)}):**")
                    # Display in columns for better layout
                    cols = st.columns(3)
                    for idx, feature in enumerate(available_features):
                        with cols[idx % 3]:
                            # Add data type and sample stats
                            if feature in processed_data.columns:
                                if processed_data[feature].dtype in ['int64', 'float64']:
                                    mean_val = processed_data[feature].mean()
                                    std_val = processed_data[feature].std()
                                    st.markdown(f"• **{feature}**: μ={mean_val:.2f}, σ={std_val:.2f}")
                                else:
                                    unique_count = processed_data[feature].nunique()
                                    st.markdown(f"• **{feature}**: {unique_count} unique values")

                if missing_features:
                    st.markdown(f"**❌ Missing Features ({len(missing_features)}):**")
                    st.markdown(", ".join(missing_features))

        # Height vs Age Analysis
        st.markdown("### 📏 Height vs Age Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Create height vs age scatterplot
            if 'Height_mm' in processed_data.columns and 'Age' in processed_data.columns:
                # Convert height to meters for better readability
                if 'Height_m' not in processed_data.columns:
                    processed_data['Height_m'] = processed_data['Height_mm'] / 1000

                fig_height_age = px.scatter(
                    processed_data,
                    x='Age',
                    y='Height_m',
                    color='AgeCategory' if 'AgeCategory' in processed_data.columns else None,
                    size='BodyMass_kg' if 'BodyMass_kg' in processed_data.columns else None,
                    hover_data=['Sex', 'BMI'] if all(col in processed_data.columns for col in ['Sex', 'BMI']) else None,
                    title="Subject Height vs Age",
                    color_discrete_map=PROFESSIONAL_COLORS['age_categories'] if 'AgeCategory' in processed_data.columns else None,
                    trendline="ols"
                )

                fig_height_age.update_layout(
                    xaxis_title="Age (years)",
                    yaxis_title="Height (m)",
                    height=400
                )

                st.plotly_chart(fig_height_age, use_container_width=True)
            else:
                st.error("Height and/or Age data not available for analysis")

        with col2:
            # Height statistics by age category
            st.markdown("#### Height Statistics")

            if 'Height_m' in processed_data.columns and 'AgeCategory' in processed_data.columns:
                height_stats = processed_data.groupby('AgeCategory')['Height_m'].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(3)
                height_stats.columns = ['N', 'Mean (m)', 'Std (m)', 'Min (m)', 'Max (m)']
                st.dataframe(height_stats)

                # Age-height correlation
                if len(processed_data[['Age', 'Height_m']].dropna()) > 1:
                    from scipy.stats import pearsonr
                    height_age_data = processed_data[['Age', 'Height_m']].dropna()
                    corr_coef, corr_p = pearsonr(height_age_data['Age'], height_age_data['Height_m'])
                    st.markdown(f"**Age-Height Correlation:**")
                    st.markdown(f"r = {corr_coef:.3f} (p = {corr_p:.3f})")

                    # Interpretation
                    if corr_p < 0.05:
                        if abs(corr_coef) < 0.1:
                            interpretation = "negligible"
                        elif abs(corr_coef) < 0.3:
                            interpretation = "weak"
                        elif abs(corr_coef) < 0.5:
                            interpretation = "moderate"
                        else:
                            interpretation = "strong"

                        direction = "positive" if corr_coef > 0 else "negative"
                        st.markdown(f"*{interpretation.title()} {direction} correlation*")
                    else:
                        st.markdown("*No significant correlation*")
            else:
                st.error("Height or age category data not available")

    # ============================================================================
    # TAB 2: SPATIOTEMPORAL ANALYSIS
    # ============================================================================
    with tab2:
        st.markdown("## 🚶‍♂️ Spatiotemporal Gait Analysis")

        if 'AvgSpeed_mps' not in filtered_data.columns:
            st.error("Walking speed data not available in the dataset")
            return

        col1, col2 = st.columns(2)

        with col1:
            # Walking speed by age group
            fig_speed_box = px.box(
                filtered_data,
                x='AgeCategory',
                y='AvgSpeed_mps',
                title="Walking Speed Distribution by Age Group",
                color='AgeCategory',
                color_discrete_map=PROFESSIONAL_COLORS['age_categories']
            )

            # Add clinical thresholds
            if show_clinical_1_0:
                fig_speed_box.add_hline(y=1.0, line_dash="dash", line_color="red",
                                      annotation_text="Clinical Threshold (1.0 m/s)")
            if show_clinical_0_8:
                fig_speed_box.add_hline(y=0.8, line_dash="dash", line_color="darkred",
                                      annotation_text="High Risk Threshold (0.8 m/s)")

            fig_speed_box.update_layout(height=400)
            st.plotly_chart(fig_speed_box, use_container_width=True)

            # Calculate and display statistics
            stats, f_stat, p_val = calculate_statistics(filtered_data, 'AgeCategory', 'AvgSpeed_mps')

            st.markdown("#### Walking Speed Statistics")
            for category, stat in stats.items():
                risk_below_1_0 = (filtered_data[filtered_data['AgeCategory'] == category]['AvgSpeed_mps'] < 1.0).sum()
                total_in_category = len(filtered_data[filtered_data['AgeCategory'] == category])
                risk_pct = (risk_below_1_0 / total_in_category * 100) if total_in_category > 0 else 0

                st.markdown(f"""
                **{category}**: {stat['mean']:.2f} ± {stat['std']:.2f} m/s
                *Risk rate*: {risk_below_1_0}/{total_in_category} ({risk_pct:.1f}%) below 1.0 m/s
                """)

            if not np.isnan(p_val) and p_val < 0.05:
                st.markdown(f'<span class="significance-badge">Significant difference (p={p_val:.3f})</span>', unsafe_allow_html=True)

        with col2:
            # Age vs speed scatter with trend
            fig_scatter = px.scatter(
                filtered_data,
                x='Age',
                y='AvgSpeed_mps',
                color='AgeCategory',
                title="Walking Speed vs Age with Trend Line",
                trendline="ols",
                color_discrete_map=PROFESSIONAL_COLORS['age_categories']
            )

            # Add clinical thresholds
            if show_clinical_1_0:
                fig_scatter.add_hline(y=1.0, line_dash="dash", line_color="red")
            if show_clinical_0_8:
                fig_scatter.add_hline(y=0.8, line_dash="dash", line_color="darkred")

            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Correlation analysis
            try:
                corr_coef, corr_p = pearsonr(filtered_data['Age'].dropna(), filtered_data['AvgSpeed_mps'].dropna())

                st.markdown("#### Age-Speed Correlation")
                st.markdown(f"""
                **Pearson r**: {corr_coef:.3f}
                **p-value**: {corr_p:.3f}
                **Interpretation**: {'Strong' if abs(corr_coef) > 0.7 else 'Moderate' if abs(corr_coef) > 0.3 else 'Weak'} {'negative' if corr_coef < 0 else 'positive'} correlation
                """)
            except Exception as e:
                st.error(f"Could not calculate correlation: {e}")

        # Clinical risk analysis
        st.markdown("### 📊 Clinical Risk Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Overall distribution
            fig_dist = px.histogram(
                filtered_data,
                x='AvgSpeed_mps',
                nbins=25,
                title="Overall Speed Distribution",
                marginal="box"
            )
            fig_dist.update_layout(height=350)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            # By age category
            fig_dist_age = px.violin(
                filtered_data,
                x='AgeCategory',
                y='AvgSpeed_mps',
                title="Speed Distribution by Age Group",
                color='AgeCategory',
                color_discrete_map=PROFESSIONAL_COLORS['age_categories']
            )
            fig_dist_age.update_layout(height=350)
            st.plotly_chart(fig_dist_age, use_container_width=True)

        with col3:
            # Clinical risk summary
            total_subjects = len(filtered_data)
            below_1_0 = (filtered_data['AvgSpeed_mps'] < 1.0).sum()
            below_0_8 = (filtered_data['AvgSpeed_mps'] < 0.8).sum()

            st.markdown("#### Clinical Risk Summary")
            create_kpi_card(f"{below_1_0}/{total_subjects}", f"Below 1.0 m/s ({below_1_0/total_subjects*100:.1f}%)", "warning")
            create_kpi_card(f"{below_0_8}/{total_subjects}", f"Below 0.8 m/s ({below_0_8/total_subjects*100:.1f}%)", "danger")

    # ============================================================================
    # TAB 3: JOINT ROM ANALYSIS
    # ============================================================================
    with tab3:
        st.markdown("## 🦵 Joint Range of Motion Analysis")

        # Joint ROM variables
        rom_vars = [col for col in ['Ankle_ROM', 'Knee_ROM', 'Hip_ROM'] if col in filtered_data.columns]

        if not rom_vars:
            st.error("ROM data not available in the dataset")
            return

        # Create subplot for all joints
        fig_rom = make_subplots(
            rows=1, cols=len(rom_vars),
            subplot_titles=[var.replace('_', ' ') for var in rom_vars],
            shared_yaxes=False
        )

        colors = PROFESSIONAL_COLORS['age_categories']

        for i, joint in enumerate(rom_vars):
            for j, age_cat in enumerate(filtered_data['AgeCategory'].unique()):
                data_subset = filtered_data[filtered_data['AgeCategory'] == age_cat][joint].dropna()

                fig_rom.add_trace(
                    go.Box(
                        y=data_subset,
                        name=age_cat,
                        marker_color=colors.get(age_cat, '#1f77b4'),
                        showlegend=(i == 0),  # Only show legend for first subplot
                        legendgroup=age_cat
                    ),
                    row=1, col=i+1
                )

        fig_rom.update_layout(height=500, title="Joint ROM Comparison Across Age Groups")
        fig_rom.update_yaxes(title_text="ROM (degrees)")
        st.plotly_chart(fig_rom, use_container_width=True)

        # Statistical analysis for ROM
        st.markdown("### 📊 ROM Statistical Analysis")

        cols = st.columns(len(rom_vars))

        for i, joint in enumerate(rom_vars):
            with cols[i]:
                joint_name = joint.split('_')[0]

                # Calculate statistics
                stats, f_stat, p_val = calculate_statistics(filtered_data, 'AgeCategory', joint)
                effect_size = calculate_effect_size(filtered_data, 'AgeCategory', joint)

                st.markdown(f"#### {joint_name} ROM")

                # Age group means
                for category, stat in stats.items():
                    st.markdown(f"**{category}**: {stat['mean']:.1f}° ± {stat['std']:.1f}°")

                # Statistical significance
                if not np.isnan(p_val) and p_val < 0.05:
                    st.markdown(f'<span class="significance-badge">p={p_val:.3f}</span>', unsafe_allow_html=True)
                    if not np.isnan(effect_size):
                        st.markdown(f"**Effect size (η²)**: {effect_size:.3f}")

                        # Interpret effect size
                        if effect_size > 0.14:
                            effect_interp = "Large effect"
                        elif effect_size > 0.06:
                            effect_interp = "Medium effect"
                        else:
                            effect_interp = "Small effect"

                        st.markdown(f"*{effect_interp}*")

                # Calculate Young → Senior change if both groups exist
                if 'Young' in stats and 'Senior' in stats:
                    young_mean = stats['Young']['mean']
                    senior_mean = stats['Senior']['mean']
                    pct_change = ((senior_mean - young_mean) / young_mean) * 100

                    st.markdown(f"**Young → Senior Change**: {pct_change:+.1f}%")

        # ROM trend analysis
        st.markdown("### 📈 ROM Age Trends")

        fig_trends = make_subplots(
            rows=1, cols=len(rom_vars),
            subplot_titles=[f"{var.replace('_', ' ')} vs Age" for var in rom_vars]
        )

        for i, joint in enumerate(rom_vars):
            # Scatter plot
            fig_trends.add_trace(
                go.Scatter(
                    x=filtered_data['Age'],
                    y=filtered_data[joint],
                    mode='markers',
                    name=joint.split('_')[0],
                    marker=dict(
                        color=filtered_data['Age'],
                        colorscale='viridis',
                        showscale=(i == len(rom_vars)-1)
                    ),
                    showlegend=False
                ),
                row=1, col=i+1
            )

            # Trend line
            try:
                valid_data = filtered_data[['Age', joint]].dropna()
                if len(valid_data) > 1:
                    z = np.polyfit(valid_data['Age'], valid_data[joint], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(valid_data['Age'].min(), valid_data['Age'].max(), 100)

                    fig_trends.add_trace(
                        go.Scatter(
                            x=x_trend,
                            y=p(x_trend),
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name=f'{joint.split("_")[0]} Trend',
                            showlegend=False
                        ),
                        row=1, col=i+1
                    )
            except:
                pass  # Skip if trend line calculation fails

        fig_trends.update_layout(height=400, title="ROM vs Age with Trend Lines")
        fig_trends.update_xaxes(title_text="Age (years)")
        fig_trends.update_yaxes(title_text="ROM (degrees)")
        st.plotly_chart(fig_trends, use_container_width=True)

    # ============================================================================
    # TAB 4: JOINT MOMENTS
    # ============================================================================
    with tab4:
        st.markdown("## 💪 Joint Moments Analysis")

        moment_vars = [col for col in ['Ankle_Peak_Moment', 'Knee_Peak_Moment', 'Hip_Peak_Moment']
                      if col in filtered_data.columns]

        if not moment_vars:
            st.error("Joint moment data not available in the dataset")
            return

        # Box plots for moments
        rows = 2
        cols = 2
        fig_moments = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[var.replace('_', ' ') for var in moment_vars] + ['Moment Comparison'],
            specs=[[{}, {}], [{}, {}]]
        )

        colors = PROFESSIONAL_COLORS['age_categories']

        # Individual joint plots
        for i, joint in enumerate(moment_vars[:3]):  # Limit to first 3 joints
            row = i // 2 + 1
            col = i % 2 + 1

            for age_cat in filtered_data['AgeCategory'].unique():
                data_subset = filtered_data[filtered_data['AgeCategory'] == age_cat][joint].dropna()

                fig_moments.add_trace(
                    go.Box(
                        y=data_subset,
                        name=age_cat,
                        marker_color=colors.get(age_cat, '#1f77b4'),
                        showlegend=(i == 0),
                        legendgroup=age_cat
                    ),
                    row=row, col=col
                )

        # Comparison plot (all joints)
        if len(moment_vars) > 0:
            moment_data_melted = pd.melt(
                filtered_data[moment_vars + ['AgeCategory']],
                id_vars=['AgeCategory'],
                var_name='Joint',
                value_name='Peak_Moment'
            )
            moment_data_melted['Joint'] = moment_data_melted['Joint'].str.replace('_Peak_Moment', '')

            for joint in moment_data_melted['Joint'].unique():
                joint_data = moment_data_melted[moment_data_melted['Joint'] == joint]
                fig_moments.add_trace(
                    go.Box(
                        x=joint_data['AgeCategory'],
                        y=joint_data['Peak_Moment'],
                        name=joint,
                        showlegend=False
                    ),
                    row=2, col=2
                )

        fig_moments.update_layout(height=600, title="Joint Peak Moments Analysis")
        fig_moments.update_yaxes(title_text="Peak Moment (Nm/kg)")
        st.plotly_chart(fig_moments, use_container_width=True)

        # Statistical analysis
        st.markdown("### 📊 Moment Statistics & Clinical Interpretation")

        cols = st.columns(len(moment_vars))

        joint_names = [var.split('_')[0] for var in moment_vars]
        for i, (joint_var, joint_name) in enumerate(zip(moment_vars, joint_names)):
            with cols[i]:
                st.markdown(f"#### {joint_name} Peak Moment")

                # Statistics
                stats, f_stat, p_val = calculate_statistics(filtered_data, 'AgeCategory', joint_var)
                effect_size = calculate_effect_size(filtered_data, 'AgeCategory', joint_var)

                # Display means by age group
                for category, stat in stats.items():
                    st.markdown(f"**{category}**: {stat['mean']:.2f} ± {stat['std']:.2f} Nm/kg")

                # Significance and effect size
                if not np.isnan(p_val) and p_val < 0.05:
                    st.markdown(f'<span class="significance-badge">p={p_val:.3f}</span>', unsafe_allow_html=True)
                    if not np.isnan(effect_size):
                        st.markdown(f"**η²**: {effect_size:.3f}")

                # Age correlation
                try:
                    valid_data = filtered_data[['Age', joint_var]].dropna()
                    if len(valid_data) > 1:
                        corr_coef, corr_p = pearsonr(valid_data['Age'], valid_data[joint_var])
                        st.markdown(f"**Age correlation**: r={corr_coef:.3f} (p={corr_p:.3f})")
                except:
                    pass

                # Young → Senior change
                if 'Young' in stats and 'Senior' in stats:
                    young_mean = stats['Young']['mean']
                    senior_mean = stats['Senior']['mean']
                    pct_change = ((senior_mean - young_mean) / young_mean) * 100

                    st.markdown(f"**Young → Senior**: {pct_change:+.1f}%")

                    # Clinical interpretation
                    if joint_name == 'Ankle' and pct_change < -10:
                        st.markdown("⚠️ *Reduced push-off power*")
                    elif joint_name == 'Hip' and pct_change > 10:
                        st.markdown("💪 *Compensatory hip strategy*")

        # Moment efficiency analysis
        efficiency_vars = [col for col in ['Ankle_Moment_Efficiency', 'Knee_Moment_Efficiency', 'Hip_Moment_Efficiency']
                          if col in filtered_data.columns]

        if efficiency_vars:
            st.markdown("### ⚡ Moment Efficiency Analysis")

            efficiency_data = pd.melt(
                filtered_data[efficiency_vars + ['AgeCategory']],
                id_vars=['AgeCategory'],
                var_name='Joint',
                value_name='Efficiency'
            )
            efficiency_data['Joint'] = efficiency_data['Joint'].str.replace('_Moment_Efficiency', '')

            fig_efficiency = px.box(
                efficiency_data,
                x='Joint',
                y='Efficiency',
                color='AgeCategory',
                title="Moment Efficiency by Joint and Age Group",
                color_discrete_map=PROFESSIONAL_COLORS['age_categories']
            )
            fig_efficiency.update_layout(height=400)
            st.plotly_chart(fig_efficiency, use_container_width=True)

    # ============================================================================
    # TAB 5: JOINT POWER
    # ============================================================================
    with tab5:
        st.markdown("## ⚡ Joint Power Analysis")

        power_vars = [col for col in ['Ankle_Peak_Power', 'Knee_Peak_Power', 'Hip_Peak_Power']
                     if col in filtered_data.columns]

        if not power_vars:
            st.error("Joint power data not available in the dataset")
            return

        # Power generation vs absorption analysis
        col1, col2 = st.columns(2)

        with col1:
            # Peak power comparison
            power_data = pd.melt(
                filtered_data[power_vars + ['AgeCategory']],
                id_vars=['AgeCategory'],
                var_name='Joint',
                value_name='Peak_Power'
            )
            power_data['Joint'] = power_data['Joint'].str.replace('_Peak_Power', '')

            fig_power = px.box(
                power_data,
                x='Joint',
                y='Peak_Power',
                color='AgeCategory',
                title="Peak Power by Joint and Age Group",
                color_discrete_map=PROFESSIONAL_COLORS['age_categories']
            )
            fig_power.update_layout(height=450)
            st.plotly_chart(fig_power, use_container_width=True)

        with col2:
            # Power vs Age scatter
            power_age_data = pd.melt(
                filtered_data[power_vars + ['Age']],
                id_vars=['Age'],
                var_name='Joint',
                value_name='Peak_Power'
            )
            power_age_data['Joint'] = power_age_data['Joint'].str.replace('_Peak_Power', '')

            fig_power_age = px.scatter(
                power_age_data,
                x='Age',
                y='Peak_Power',
                color='Joint',
                title="Peak Power vs Age by Joint",
                trendline="ols",
                facet_col='Joint'
            )
            fig_power_age.update_layout(height=450)
            st.plotly_chart(fig_power_age, use_container_width=True)

        # Power statistics
        st.markdown("### 📊 Power Generation Statistics")

        cols = st.columns(len(power_vars))
        joint_names = [var.split('_')[0] for var in power_vars]

        for i, (power_var, joint_name) in enumerate(zip(power_vars, joint_names)):
            with cols[i]:
                st.markdown(f"#### {joint_name} Peak Power")

                # Statistics
                stats, f_stat, p_val = calculate_statistics(filtered_data, 'AgeCategory', power_var)

                # Display means
                for category, stat in stats.items():
                    st.markdown(f"**{category}**: {stat['mean']:.2f} ± {stat['std']:.2f} W/kg")

                # Significance
                if not np.isnan(p_val) and p_val < 0.05:
                    st.markdown(f'<span class="significance-badge">p={p_val:.3f}</span>', unsafe_allow_html=True)

                # Young → Senior change
                if 'Young' in stats and 'Senior' in stats:
                    young_mean = stats['Young']['mean']
                    senior_mean = stats['Senior']['mean']
                    pct_change = ((senior_mean - young_mean) / young_mean) * 100

                    if pct_change < -15:
                        color = "danger"
                        interpretation = "Substantial decline"
                    elif pct_change < -5:
                        color = "warning"
                        interpretation = "Moderate decline"
                    else:
                        color = "success"
                        interpretation = "Well preserved"

                    create_kpi_card(f"{pct_change:+.1f}%", f"Young→Senior\\n{interpretation}", color)

        # Power hierarchy summary
        st.markdown("### 🏆 Joint Power Hierarchy")

        # Calculate average power by joint across all subjects
        joint_power_means = {}
        for var in power_vars:
            joint = var.split('_')[0]
            joint_power_means[joint] = filtered_data[var].mean()

        # Sort by power output
        sorted_joints = sorted(joint_power_means.items(), key=lambda x: x[1], reverse=True)

        cols = st.columns(len(sorted_joints))

        for i, (joint, power) in enumerate(sorted_joints):
            with cols[i]:
                rank_colors = PROFESSIONAL_COLORS['rankings']  # Professional ranking colors
                color = rank_colors[i] if i < 3 else "#95a5a6"

                st.markdown(f"""
                <div style="background: {color}; color: black; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold;">#{i+1} {joint}</div>
                    <div style="font-size: 1.2rem; margin-top: 0.25rem;">{power:.2f} W/kg</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### Clinical Insights")
        st.markdown("""
        - **Ankle power decline** is typically the earliest and most pronounced age-related change
        - **Hip compensation** often increases to maintain walking speed
        - **Knee power preservation** varies significantly between individuals
        """)

    # ============================================================================
    # TAB 6: GAIT CYCLE EXPLORER
    # ============================================================================
    with tab6:
        st.markdown("## 🔍 Gait Cycle Explorer")
        st.markdown("### Explore individual subject gait patterns throughout the complete gait cycle")

        # Load MAT data function
        @st.cache_data
        def load_mat_subjects_list():
            """Load list of available subjects from MAT file"""
            mat_file = 'data/processed/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx'
            main_df = pd.read_csv('data/processed/final_advanced_biomechanical_dataset.csv')

            try:
                excel_file = pd.ExcelFile(mat_file)
                subjects = [sheet for sheet in excel_file.sheet_names if sheet.startswith('Sub')]

                subject_info = []
                for subject_sheet in subjects:
                    subject_num = int(subject_sheet[3:])
                    main_subject_data = main_df[main_df['ID'] == f'SUBJ{subject_num}']

                    if not main_subject_data.empty:
                        subject_data = main_subject_data.iloc[0]
                        subject_info.append({
                            'sheet': subject_sheet,
                            'id': f'SUBJ{subject_num}',
                            'age': subject_data['Age'],
                            'age_group': subject_data['AgeCategory'],
                            'sex': subject_data['Sex'],
                            'speed': subject_data['AvgSpeed_mps']
                        })

                return subject_info
            except Exception as e:
                st.error(f"Error loading subjects: {e}")
                return []

        @st.cache_data
        def load_subject_timeseries(subject_sheet):
            """Load time-series data for a specific subject"""
            mat_file = 'data/processed/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx'
            try:
                ts_df = pd.read_excel(mat_file, sheet_name=subject_sheet)
                return ts_df
            except Exception as e:
                st.error(f"Error loading subject {subject_sheet}: {e}")
                return None

        @st.cache_data
        def calculate_age_group_means():
            """Calculate mean curves for each age group"""
            mat_file = 'data/processed/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx'
            main_df = pd.read_csv('data/processed/final_advanced_biomechanical_dataset.csv')

            try:
                excel_file = pd.ExcelFile(mat_file)
                subjects = [sheet for sheet in excel_file.sheet_names if sheet.startswith('Sub')]

                # Get variable list
                sample_df = pd.read_excel(mat_file, sheet_name='Sub01')
                variables = sample_df.columns.tolist()

                age_group_data = {'Young': {}, 'Adult': {}, 'Senior': {}}

                # Initialize lists for each variable and age group
                for age_group in age_group_data:
                    for var in variables:
                        age_group_data[age_group][var] = []

                # Load data by age group
                for subject_sheet in subjects:
                    subject_num = int(subject_sheet[3:])
                    main_subject_data = main_df[main_df['ID'] == f'SUBJ{subject_num}']

                    if not main_subject_data.empty:
                        ts_df = pd.read_excel(mat_file, sheet_name=subject_sheet)
                        subject_info = main_subject_data.iloc[0]
                        age_group = subject_info['AgeCategory']

                        for var in variables:
                            if var in ts_df.columns:
                                age_group_data[age_group][var].append(ts_df[var].values)

                # Calculate means
                age_group_means = {}
                for age_group in age_group_data:
                    age_group_means[age_group] = {}
                    for var in variables:
                        if age_group_data[age_group][var]:
                            age_group_means[age_group][var] = np.mean(age_group_data[age_group][var], axis=0)
                        else:
                            age_group_means[age_group][var] = None

                return age_group_means, variables
            except Exception as e:
                st.error(f"Error calculating age group means: {e}")
                return {}, []

        # Load subjects and variables
        subjects_info = load_mat_subjects_list()
        age_group_means, available_variables = calculate_age_group_means()

        if not subjects_info:
            st.error("No subjects loaded. Please check the MAT file.")
            return

        # Controls section
        col1, col2 = st.columns(2)

        with col1:
            # Subject selection with enhanced info
            subject_options = []
            for subject in subjects_info:
                label = f"{subject['id']} (Age: {subject['age']}, {subject['age_group']}, {subject['sex']}, Speed: {subject['speed']:.2f} m/s)"
                subject_options.append((label, subject))

            selected_subject_idx = st.selectbox(
                "Select Subject",
                range(len(subject_options)),
                format_func=lambda x: subject_options[x][0]
            )

            selected_subject = subject_options[selected_subject_idx][1] if subject_options else None

        with col2:
            # Variable selection organized by category
            variable_categories = {
                "Joint Kinematics": [v for v in available_variables if 'Angles' in v],
                "Joint Kinetics": [v for v in available_variables if 'Moment' in v],
                "Joint Power": [v for v in available_variables if 'Power' in v],
                "Muscle Activity (EMG)": [v for v in available_variables if 'norm' in v],
                "Ground Reaction Forces": [v for v in available_variables if 'GRF_' in v]
            }

            category = st.selectbox("Variable Category", list(variable_categories.keys()))
            if variable_categories[category]:
                selected_variables = st.multiselect(
                    f"Select {category} Variables",
                    variable_categories[category],
                    default=variable_categories[category][:2] if len(variable_categories[category]) >= 2 else variable_categories[category][:1]
                )
            else:
                selected_variables = []
                st.warning(f"No variables found for {category}")

        # Options
        col3, col4 = st.columns(2)
        with col3:
            show_age_groups = st.checkbox("Compare with Age Group Means", value=True)
        with col4:
            normalize_data = st.checkbox("Normalize to [0,1] range", value=False)

        if not selected_subject or not selected_variables:
            st.info("Please select a subject and at least one variable to visualize.")
            return

        # Load subject data
        with st.spinner("Loading subject data..."):
            subject_data = load_subject_timeseries(selected_subject['sheet'])

        if subject_data is None:
            st.error("Failed to load subject data.")
            return

        # Create visualization
        st.markdown("### 📈 Gait Cycle Visualization")

        # Subject info display
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            create_kpi_card(f"{selected_subject['age']}", "Age (years)", "primary")
        with col_info2:
            create_kpi_card(selected_subject['age_group'], "Age Group", "secondary")
        with col_info3:
            create_kpi_card(selected_subject['sex'], "Sex", "info")
        with col_info4:
            create_kpi_card(f"{selected_subject['speed']:.3f}", "Speed (m/s)", "success")

        # Create subplot for each selected variable
        n_vars = len(selected_variables)
        if n_vars == 1:
            rows, cols = 1, 1
        elif n_vars == 2:
            rows, cols = 1, 2
        elif n_vars <= 4:
            rows, cols = 2, 2
        elif n_vars <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3  # Max 9 variables

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=selected_variables,
            vertical_spacing=0.08,
            horizontal_spacing=0.06
        )

        # Color scheme
        colors = {
            'subject': PROFESSIONAL_COLORS['primary'],  # Primary color for subject
            **PROFESSIONAL_COLORS['age_categories']     # Age category colors
        }

        gait_cycle = np.linspace(0, 100, 1001)  # 0-100% gait cycle

        for i, variable in enumerate(selected_variables):
            row = (i // cols) + 1
            col = (i % cols) + 1

            if variable in subject_data.columns:
                # Get subject data
                y_data = subject_data[variable].values

                # Normalize if requested
                if normalize_data:
                    y_min, y_max = y_data.min(), y_data.max()
                    if y_max > y_min:
                        y_data = (y_data - y_min) / (y_max - y_min)

                # Plot subject curve
                fig.add_trace(
                    go.Scatter(
                        x=gait_cycle,
                        y=y_data,
                        name=f"{selected_subject['id']} - {variable}",
                        line=dict(color=colors['subject'], width=3),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )

                # Add age group means if requested
                if show_age_groups and age_group_means:
                    for age_group in ['Young', 'Adult', 'Senior']:
                        if (age_group in age_group_means and
                            variable in age_group_means[age_group] and
                            age_group_means[age_group][variable] is not None):

                            mean_data = age_group_means[age_group][variable]

                            # Normalize if requested
                            if normalize_data:
                                if y_max > y_min:  # Use same normalization as subject
                                    mean_data = (mean_data - y_min) / (y_max - y_min)

                            # Add transparency for age group means
                            alpha = 0.7
                            line_style = 'dash' if age_group != selected_subject['age_group'] else 'solid'
                            line_width = 3 if age_group == selected_subject['age_group'] else 2

                            fig.add_trace(
                                go.Scatter(
                                    x=gait_cycle,
                                    y=mean_data,
                                    name=f"{age_group} Mean",
                                    line=dict(
                                        color=colors[age_group],
                                        width=line_width,
                                        dash=line_style
                                    ),
                                    opacity=alpha,
                                    showlegend=(i == 0)  # Only show legend for first subplot
                                ),
                                row=row, col=col
                            )

                # Update axis labels
                fig.update_xaxes(title_text="Gait Cycle (%)", row=row, col=col)

                if normalize_data:
                    fig.update_yaxes(title_text=f"{variable} (Normalized)", row=row, col=col)
                else:
                    # Get units from variable name
                    if 'Angles' in variable:
                        unit = "(degrees)"
                    elif 'Moment' in variable:
                        unit = "(Nm/kg)"
                    elif 'Power' in variable:
                        unit = "(W/kg)"
                    elif 'GRF_' in variable:
                        unit = "(N/kg)"
                    elif 'norm' in variable:
                        unit = "(normalized)"
                    else:
                        unit = ""

                    fig.update_yaxes(title_text=f"{variable} {unit}", row=row, col=col)

        # Update layout
        title_text = f"Gait Cycle Analysis: {selected_subject['id']} ({selected_subject['age_group']}, {selected_subject['age']}y)"
        fig.update_layout(
            title=title_text,
            height=300 * rows,  # Dynamic height based on number of rows
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistical summary
        if show_age_groups:
            st.markdown("### 📊 Comparison Summary")

            summary_data = []
            for variable in selected_variables:
                if variable in subject_data.columns:
                    subject_mean = subject_data[variable].mean()
                    subject_std = subject_data[variable].std()

                    row_data = {
                        'Variable': variable,
                        f'{selected_subject["id"]} Mean': f'{subject_mean:.3f}',
                        f'{selected_subject["id"]} SD': f'{subject_std:.3f}'
                    }

                    # Add age group comparisons
                    for age_group in ['Young', 'Adult', 'Senior']:
                        if (age_group in age_group_means and
                            variable in age_group_means[age_group] and
                            age_group_means[age_group][variable] is not None):

                            group_mean = np.mean(age_group_means[age_group][variable])
                            row_data[f'{age_group} Mean'] = f'{group_mean:.3f}'

                            # Calculate difference from group mean
                            diff = subject_mean - group_mean
                            row_data[f'Diff from {age_group}'] = f'{diff:+.3f}'

                    summary_data.append(row_data)

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

        # Data export section
        st.markdown("### 💾 Export Data")
        col_export1, col_export2 = st.columns(2)

        with col_export1:
            if st.button("📥 Download Subject Data"):
                # Prepare CSV data
                export_data = subject_data[selected_variables].copy()
                export_data.insert(0, 'Gait_Cycle_Percent', gait_cycle)

                csv = export_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{selected_subject["id"]}_gait_data.csv">Download Subject Data CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

        with col_export2:
            if st.button("🖼️ Download Plot"):
                # Export plot as HTML
                plot_html = fig.to_html(include_plotlyjs='cdn')
                b64 = base64.b64encode(plot_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="{selected_subject["id"]}_gait_plot.html">Download Interactive Plot</a>'
                st.markdown(href, unsafe_allow_html=True)

        # Additional insights
        st.markdown("### 🔍 Key Insights")
        insights_col1, insights_col2 = st.columns(2)

        with insights_col1:
            st.markdown(f"""
            **Subject Profile:**
            - **{selected_subject['id']}**: {selected_subject['age']}-year-old {selected_subject['sex']}
            - **Age Group**: {selected_subject['age_group']}
            - **Walking Speed**: {selected_subject['speed']:.3f} m/s
            """)

        with insights_col2:
            if show_age_groups and selected_variables:
                variable = selected_variables[0]  # Use first selected variable
                if (variable in subject_data.columns and
                    selected_subject['age_group'] in age_group_means and
                    variable in age_group_means[selected_subject['age_group']] and
                    age_group_means[selected_subject['age_group']][variable] is not None):

                    subject_mean = subject_data[variable].mean()
                    group_mean = np.mean(age_group_means[selected_subject['age_group']][variable])

                    if abs(subject_mean - group_mean) / group_mean > 0.1:
                        trend = "above" if subject_mean > group_mean else "below"
                        st.markdown(f"""
                        **Pattern Analysis:**
                        - Subject shows {variable} values {trend} their age group average
                        - Difference: {((subject_mean - group_mean) / group_mean * 100):+.1f}%
                        - This may indicate unique gait characteristics
                        """)
                    else:
                        st.markdown(f"""
                        **Pattern Analysis:**
                        - Subject shows typical {variable} patterns for their age group
                        - Values are within normal range (±10% of group mean)
                        - Consistent with expected {selected_subject['age_group']} gait patterns
                        """)
    # ============================================================================
    # TAB 7: SUBJECT EXPLORER
    # ============================================================================
    with tab7:
        st.markdown("## 👤 Individual Subject Explorer")

        # Subject selection
        subject_list = sorted(filtered_data['ID'].tolist())
        selected_subject = st.selectbox("Select Subject", subject_list)

        if selected_subject:
            subject_data = filtered_data[filtered_data['ID'] == selected_subject].iloc[0]

            # Subject overview
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                create_kpi_card(f"{subject_data['Age']:.0f}y", "Age", "primary")
            with col2:
                create_kpi_card(subject_data['Sex'], "Sex", "success")
            with col3:
                create_kpi_card(subject_data['AgeCategory'], "Age Group", "warning")
            with col4:
                if 'BMI' in subject_data.index:
                    create_kpi_card(f"{subject_data['BMI']:.1f}", "BMI kg/m²", "primary")

            # Detailed subject profile
            st.markdown("### 📊 Subject Biomechanical Profile")

            col1, col2 = st.columns(2)

            with col1:
                # Spatiotemporal metrics
                st.markdown("#### 🚶‍♂️ Spatiotemporal")
                spatio_metrics = {}

                if 'AvgSpeed_mps' in subject_data.index:
                    spatio_metrics['Walking Speed'] = f"{subject_data['AvgSpeed_mps']:.2f} m/s"
                if 'SpeedCategory' in subject_data.index:
                    spatio_metrics['Speed Category'] = subject_data['SpeedCategory']
                if 'FroudeNumber' in subject_data.index:
                    spatio_metrics['Froude Number'] = f"{subject_data['FroudeNumber']:.3f}"
                if 'GaitEfficiency' in subject_data.index:
                    spatio_metrics['Gait Efficiency'] = f"{subject_data['GaitEfficiency']:.3f}"
                if 'SpeedAsymmetry_pct' in subject_data.index:
                    spatio_metrics['Speed Asymmetry'] = f"{subject_data['SpeedAsymmetry_pct']:.1f}%"

                for metric, value in spatio_metrics.items():
                    st.markdown(f"**{metric}**: {value}")

                # Clinical flags
                if 'AvgSpeed_mps' in subject_data.index:
                    if subject_data['AvgSpeed_mps'] < 0.8:
                        st.markdown('<span class="clinical-threshold">🚨 High fall risk (< 0.8 m/s)</span>', unsafe_allow_html=True)
                    elif subject_data['AvgSpeed_mps'] < 1.0:
                        st.markdown('<span class="clinical-threshold">⚠️ Below clinical threshold (1.0 m/s)</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="normal-threshold">✅ Normal walking speed</span>', unsafe_allow_html=True)

            with col2:
                # Joint ROM profile
                st.markdown("#### 🦵 Joint ROM Profile")
                rom_metrics = {}

                for joint in ['Ankle', 'Knee', 'Hip', 'Pelvis']:
                    rom_col = f'{joint}_ROM'
                    if rom_col in subject_data.index:
                        rom_metrics[f'{joint} ROM'] = f"{subject_data[rom_col]:.1f}°"

                for joint, value in rom_metrics.items():
                    st.markdown(f"**{joint}**: {value}")

                # ROM comparison to age group mean
                age_group_data = filtered_data[filtered_data['AgeCategory'] == subject_data['AgeCategory']]

                st.markdown("#### 📊 Compared to Age Group")
                for joint in ['Ankle', 'Knee', 'Hip']:
                    rom_col = f'{joint}_ROM'
                    if rom_col in subject_data.index and rom_col in age_group_data.columns:
                        subject_value = subject_data[rom_col]
                        percentile = (age_group_data[rom_col] < subject_value).sum() / len(age_group_data) * 100
                        st.markdown(f"**{joint}**: {percentile:.0f}th percentile")

            # Subject biomechanical radar chart
            st.markdown("### 🎯 Biomechanical Radar Profile")

            # Prepare radar chart data
            radar_columns = ['AvgSpeed_mps', 'Ankle_ROM', 'Knee_ROM', 'Hip_ROM', 'Ankle_Peak_Power', 'Hip_Peak_Power']
            radar_labels = ['Walking Speed', 'Ankle ROM', 'Knee ROM', 'Hip ROM', 'Ankle Power', 'Hip Power']

            available_metrics = {}
            available_labels = []

            for col, label in zip(radar_columns, radar_labels):
                if col in subject_data.index and col in processed_data.columns:
                    # Normalize to 0-100 scale
                    col_min = processed_data[col].min()
                    col_max = processed_data[col].max()
                    if col_max > col_min:
                        normalized = (subject_data[col] - col_min) / (col_max - col_min) * 100
                        available_metrics[label] = normalized
                        available_labels.append(label)

            if len(available_metrics) >= 3:  # Need at least 3 metrics for radar chart
                # Create radar chart
                fig_radar = go.Figure()

                fig_radar.add_trace(go.Scatterpolar(
                    r=list(available_metrics.values()),
                    theta=list(available_metrics.keys()),
                    fill='toself',
                    name=selected_subject,
                    line_color='rgb(50, 171, 96)'
                ))

                # Add age group mean for comparison
                age_group_radar = {}
                for label, col in zip(radar_labels, radar_columns):
                    if col in age_group_data.columns and label in available_metrics:
                        group_mean = age_group_data[col].mean()
                        col_min = processed_data[col].min()
                        col_max = processed_data[col].max()
                        if col_max > col_min:
                            normalized = (group_mean - col_min) / (col_max - col_min) * 100
                            age_group_radar[label] = normalized

                if age_group_radar:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=list(age_group_radar.values()),
                        theta=list(age_group_radar.keys()),
                        fill='toself',
                        name=f'{subject_data["AgeCategory"]} Group Mean',
                        line_color='rgb(255, 99, 71)',
                        opacity=0.6
                    ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=True,
                    title=f"Biomechanical Profile: {selected_subject}",
                    height=500
                )

                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.warning("Insufficient data available for radar chart visualization")

    # ============================================================================
    # TAB 8: HELP & ABOUT
    # ============================================================================
    with tab8:
        st.markdown("## ℹ️ Help & About")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Research Focus")
            st.markdown("""
            This intelligence panel analyzes age-related changes in human gait biomechanics using data from 138 healthy adults (21-86 years).

            **Key Research Questions:**
            1. How do spatiotemporal parameters change with age?
            2. Which joints show the greatest age-related ROM changes?
            3. How do joint moments and power adapt with aging?
            4. What are the clinical implications for fall risk?

            **Data Source:** Nature Scientific Data publication (Camargo-Junior et al., 2024)
            """)

            st.markdown("### 🔬 Methods & Analysis")
            st.markdown("""
            **Statistical Approaches:**
            - ANOVA for group comparisons
            - Pearson/Spearman correlations
            - Effect size calculations (η²)
            - Clinical threshold analysis
            - Statistical Parametric Mapping (SPM1D)

            **Normalization Methods:**
            - Froude number normalization (Fr = v²/(L×g))
            - Body mass normalization for kinetics
            - Z-score standardization
            """)

        with col2:
            st.markdown("### ⚠️ Clinical Thresholds")
            st.markdown("""
            **Walking Speed Thresholds:**
            - **< 1.0 m/s**: Increased fall risk
            - **< 0.8 m/s**: High fall risk, mobility impairment
            - **> 1.2 m/s**: Normal community walking

            **BMI Categories:**
            - **< 18.5**: Underweight
            - **18.5-24.9**: Normal
            - **25-29.9**: Overweight
            - **≥ 30**: Obese
            """)

            st.markdown("### 📊 Data Interpretation")
            st.markdown("""
            **Effect Size Interpretation (η²):**
            - **< 0.06**: Small effect
            - **0.06-0.14**: Medium effect
            - **> 0.14**: Large effect

            **Clinical Significance:**
            - Changes > 10% considered clinically meaningful
            - Ankle power most affected by aging
            - Hip compensation strategies common
            - Speed decline ~0.5-1% per year after 60
            """)

            st.markdown("### 🔍 Data Provenance")
            st.markdown("""
            **Dataset Characteristics:**
            - 138 healthy adults (21-86 years)
            - 88 biomechanical variables
            - Laboratory-controlled gait analysis
            - 3D kinematics, kinetics, EMG, GRF
            - Quality-controlled preprocessing

            **Missing Data:**
            - EMG data: ~21-23% missing
            - Kinematic/kinetic data: Complete
            - No systematic bias identified
            """)

    # Footer with developer info
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 20px; font-size: 14px; color: #666;'>
            <p style='margin: 0;'>
                Developed by <strong>David Rodríguez Cianca</strong>
                <a href="https://www.linkedin.com/in/david-rodriguez-cianca/" target="_blank"
                   style="text-decoration: none; margin-left: 15px; background-color: #0077B5; color: white;
                          padding: 5px 10px; border-radius: 4px; font-size: 12px; font-weight: bold;
                          display: inline-block; vertical-align: middle;">
                    🔗 LinkedIn
                </a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()