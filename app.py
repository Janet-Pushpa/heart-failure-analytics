from streamlit_extras.metric_cards import style_metric_cards # For styling metric cards, if needed
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
import shap
import statsmodels.api as sm
import time
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Deep-Pulse", page_icon="❤️")

# --- Custom Dark Theme (Using CSS) ---
st.markdown(
    """
    <style>
    .main > div {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Deep-Pulse: Predictive Health Analytics")

# --- Navigation Tabs (Renamed as per new requirements) ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🩺 Diagnostics", "📈 Trend Forecasting"])


# --- Data Loading and Preprocessing (Simulated 'Heart Failure Clinical Records' Dataset) ---
@st.cache_data
def load_and_preprocess_data():
    # Simulate 'Heart Failure Clinical Records' Dataset
    np.random.seed(42)
    data_size = 1000
    df = pd.DataFrame({
        'age': np.random.randint(40, 95, data_size),
        'anaemia': np.random.choice([0, 1], data_size, p=[0.6, 0.4]),
        'creatinine_phosphokinase': np.random.randint(20, 800, data_size),
        'diabetes': np.random.choice([0, 1], data_size, p=[0.6, 0.4]),
        'ejection_fraction': np.random.randint(20, 80, data_size),
        'high_blood_pressure': np.random.choice([0, 1], data_size, p=[0.7, 0.3]),
        'platelets': np.random.randint(150000, 500000, data_size),
        'serum_creatinine': np.round(np.random.uniform(0.5, 9.0, data_size), 2), # New feature
        'serum_sodium': np.random.randint(110, 150, data_size),
        'sex': np.random.choice([0, 1], data_size, p=[0.5, 0.5]), # 0=Female, 1=Male
        'smoking': np.random.choice([0, 1], data_size, p=[0.7, 0.3]),
        'time': np.random.randint(4, 300, data_size),
        'DEATH_EVENT': np.random.choice([0, 1], data_size, p=[0.7, 0.3]) # Target variable
    })

    # Introduce some missing values and handle them (Module 2)
    for col in ['creatinine_phosphokinase', 'platelets', 'serum_creatinine']:
        df.loc[df.sample(frac=0.03).index, col] = np.nan
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Feature Engineering (Example: Ratio of Creatinine to Age)
    df['creatinine_age_ratio'] = df['serum_creatinine'] / df['age']

    # Scaling numerical features (Module 2)
    numerical_cols = [
        'age', 'creatinine_phosphokinase', 'ejection_fraction', 
        'platelets', 'serum_creatinine', 'serum_sodium', 'time', 
        'creatinine_age_ratio'
    ]
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler

df, scaler = load_and_preprocess_data()

# Prepare data for Classification (Predicting 'DEATH_EVENT', Module 6)
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE for class imbalance (Module 6)
# SMOTE (Synthetic Minority Over-sampling Technique) works by creating synthetic samples of the minority class.
# It selects k-nearest neighbors for each minority sample, then randomly picks one neighbor
# and creates a new synthetic sample somewhere along the line segment connecting the original sample and the chosen neighbor.
# This helps to balance the class distribution without simply duplicating existing samples.
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train Random Forest Classifier (Module 6)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_smote, y_train_smote)

# Train Linear Regression for 'Serum Creatinine' prediction (Module 4)
# Using other numerical features to predict SerumCreatinine
X_serum_creatinine = df[[
    'age', 'creatinine_phosphokinase', 'ejection_fraction', 
    'platelets', 'serum_sodium', 'time'
]]
y_serum_creatinine = df['serum_creatinine']

X_train_serum, X_test_serum, y_train_serum, y_test_serum = train_test_split(X_serum_creatinine, y_serum_creatinine, test_size=0.2, random_state=42)

lin_reg_serum = LinearRegression()
lin_reg_serum.fit(X_train_serum, y_train_serum)

# --- Tab 1: Overview (Visualization) ---
with tab1:
    st.header("📊 Overview: Patient Vitals Visualization")

    st.subheader("3D Scatter Plot of Patient Vitals")
    # Using simulated 'ejection_fraction', 'serum_creatinine', 'age' for 3D plot
    fig_3d = px.scatter_3d(df.sample(200), x='age', y='serum_creatinine', z='ejection_fraction', color='DEATH_EVENT',
                           color_continuous_scale=px.colors.sequential.Viridis,
                           title="Patient Vitals (Age, Serum Creatinine, Ejection Fraction)",
                           height=600)
    fig_3d.update_layout(scene=dict(
        xaxis_title='Age (Scaled)',
        yaxis_title='Serum Creatinine (Scaled)',
        zaxis_title='Ejection Fraction (Scaled)'
    ), margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_3d, use_container_width=True)

    st.subheader("Real-time Pulse Simulation")
    pulse_placeholder = st.empty()
    t = np.linspace(0, 2 * np.pi, 100)
    for i in range(50): # Reduced iterations for faster loading/demo
        # Simulate a sine wave for pulse
        pulse = 60 + 20 * np.sin(t + i * 0.1)
        pulse_data = pd.DataFrame({'Time': t, 'Pulse': pulse})

        fig_pulse = px.line(pulse_data, x='Time', y='Pulse', title="Simulated Real-time Pulse",
                              labels={'Pulse': 'Heart Rate (BPM)', 'Time': 'Time (s)'})
        fig_pulse.update_yaxes(range=[40, 100]) # Keep y-axis consistent
        fig_pulse.update_layout(showlegend=False, margin=dict(l=0, r=0, b=0, t=40))
        pulse_placeholder.plotly_chart(fig_pulse, use_container_width=True)
        time.sleep(0.1)


# --- Tab 2: Diagnostics (Classification) ---
with tab2:
    st.header("🩺 Diagnostics: Heart Failure Risk")

    st.sidebar.header("Patient Input Features")

    age_input = st.sidebar.slider("Age", min_value=40, max_value=95, value=60)
    cpk_input = st.sidebar.slider("Creatinine Phosphokinase", min_value=20, max_value=800, value=200)
    ef_input = st.sidebar.slider("Ejection Fraction", min_value=20, max_value=80, value=35)
    platelets_input = st.sidebar.slider("Platelets (kiloplatelets/mL)", min_value=150000, max_value=500000, value=250000)
    sc_input = st.sidebar.slider("Serum Creatinine (mg/dL)", min_value=0.5, max_value=9.0, value=1.5, step=0.1)
    ss_input = st.sidebar.slider("Serum Sodium (mEq/L)", min_value=110, max_value=150, value=135)
    time_input = st.sidebar.slider("Follow-up Period (days)", min_value=4, max_value=300, value=150)

    anaemia_input = st.sidebar.checkbox("Anaemia")
    diabetes_input = st.sidebar.checkbox("Diabetes")
    hbp_input = st.sidebar.checkbox("High Blood Pressure")
    sex_input = st.sidebar.radio("Sex", ["Male", "Female"])
    smoking_input = st.sidebar.checkbox("Smoking")

    # Create a DataFrame for the current input
    input_df = pd.DataFrame({
        'age': [age_input],
        'anaemia': [1 if anaemia_input else 0],
        'creatinine_phosphokinase': [cpk_input],
        'diabetes': [1 if diabetes_input else 0],
        'ejection_fraction': [ef_input],
        'high_blood_pressure': [1 if hbp_input else 0],
        'platelets': [platelets_input],
        'serum_creatinine': [sc_input],
        'serum_sodium': [ss_input],
        'sex': [1 if sex_input == 'Male' else 0],
        'smoking': [1 if smoking_input else 0],
        'time': [time_input],
    })

    # Feature Engineering for input_df
    input_df['creatinine_age_ratio'] = input_df['serum_creatinine'] / input_df['age']

    # Ensure all columns are present and in the correct order for prediction
    input_df = input_df[X.columns] 

    # Scale the numerical input features
    numerical_cols_for_scaling = [
        'age', 'creatinine_phosphokinase', 'ejection_fraction', 
        'platelets', 'serum_creatinine', 'serum_sodium', 'time', 
        'creatinine_age_ratio'
    ]
    input_df[numerical_cols_for_scaling] = scaler.transform(input_df[numerical_cols_for_scaling])

    if st.sidebar.button("Predict Death Event Risk"):
        prediction_proba = rf_classifier.predict_proba(input_df)[:, 1][0]
        
        # Determine color based on risk levels
        card_color = "#00FF00" # Green
        if prediction_proba * 100 > 70:
            card_color = "#FF0000" # Red
        elif prediction_proba * 100 > 40:
            card_color = "#FFA500" # Orange
        elif prediction_proba * 100 > 20:
            card_color = "#FFFF00" # Yellow

        # Display the Metric Card
        st.metric(
            label="Predicted Probability of Death Event", 
            value=f"{prediction_proba * 100:.2f}%", 
            help="Probability based on Random Forest Classifier trained with SMOTE."
        )

        # This is what makes the card glow and use your 'card_color'
        style_metric_cards(
            background_color="#1f2630",
            border_left_color=card_color,
            border_size_px=5,
            box_shadow=True
        )

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction_proba * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {"text": "Risk Assessment Gauge"},
            gauge = {
                "axis": {"range": [None, 100]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, 20], "color": "green"},
                    {"range": [20, 40], "color": "yellow"},
                    {"range": [40, 70], "color": "orange"},
                    {"range": [70, 100], "color": "red"}]}))
        
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("SHAP Waterfall Plot for Explainable AI (Death Event)")
        # For SHAP, we need a single prediction explanation
        st.subheader("SHAP Waterfall Plot for Explainable AI (Death Event)")
        
        # Use the newer SHAP Explainer interface which handles the indexing for us
        explainer = shap.Explainer(rf_classifier, X_train_smote)
        shap_values = explainer(input_df)

        # We take the values for the 'Death Event' (index 1)
        # Random Forest in SHAP often returns [samples, features, classes]
        # We need to ensure we are passing a single 1D Explanation object
        exp = shap.Explanation(
            values=shap_values.values[0, :, 1], 
            base_values=shap_values.base_values[0, 1], 
            data=input_df.iloc[0].values, 
            feature_names=X.columns.tolist()
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False)
        st.pyplot(fig)
        plt.clf() # Clean up the figure after plotting


# --- Tab 3: Trend Forecasting (Time Series) ---
with tab3:
    st.header("📈 Trend Forecasting: Patient Pulse Timeline")

    st.subheader("Simulated 30-day Patient Pulse and 7-day Forecast")

    # Simulate time series data for 'Patient Pulse' over 30 days (hourly data)
    np.random.seed(0)
    n_pts_30d = 30 * 24 # 30 days * 24 hours/day
    dts_30d = pd.date_range(start='2023-01-01', periods=n_pts_30d, freq='h')
    
    # Base heart rate with components:
    # 1. Daily cycle (sin wave)
    # 2. Weekly cycle (sin wave)
    # 3. Noise (random normal)
    # 4. Trend (linear)
    daily_cycle = 5 * np.sin(np.linspace(0, 4 * np.pi * 30, n_pts_30d))
    weekly_cycle = 3 * np.sin(np.linspace(0, 2 * np.pi * (30/7), n_pts_30d))
    noise = np.random.normal(0, 1.5, n_pts_30d)
    trend = np.linspace(0, 10, n_pts_30d)
    
    pulse_vals = 70 + daily_cycle + weekly_cycle + noise + trend
    
    pulse_df = pd.DataFrame({'Date': dts_30d, 'Pulse': pulse_vals})
    pulse_df.set_index('Date', inplace=True)

    # Fit an ARIMA model (p, d, q) (Module 7)
    # ARIMA (AutoRegressive Integrated Moving Average) models are used for time series forecasting.
    # 'AR' (Autoregressive) refers to the use of past values in the regression equation.
    # 'I' (Integrated) refers to the use of differencing to make the time series stationary (remove trends).
    # 'MA' (Moving Average) refers to the use of past forecast errors in the regression equation.
    # The order (p, d, q) specifies the number of AR terms, differences, and MA terms, respectively.
    try:
        model_arima = sm.tsa.arima.ARIMA(pulse_df['Pulse'], order=(5,1,0))
        model_fit_arima = model_arima.fit()

        # Forecast for the next 7 days (7 days * 24 hours/day)
        f_steps = 7 * 24
        f_vals = model_fit_arima.predict(start=len(pulse_df), end=len(pulse_df) + f_steps - 1)
        f_idx = pd.date_range(start=pulse_df.index[-1] + pd.Timedelta(hours=1), periods=f_steps, freq='h')
        f_series = pd.Series(f_vals, index=f_idx)

        # Create Plotly line chart with shaded area for forecast
        fig_ts_arima = go.Figure()

        # Actual data
        fig_ts_arima.add_trace(go.Scatter(x=pulse_df.index, y=pulse_df['Pulse'], mode='lines', name='Actual Pulse', line=dict(color='deepskyblue')))

        # Forecast data
        fig_ts_arima.add_trace(go.Scatter(x=f_series.index, y=f_series, mode='lines', name='Forecasted Pulse', line=dict(color='salmon', dash='dot')))

        # Shaded area for uncertainty (simulated)
        u_bound = f_series + 2.5 
        l_bound = f_series - 2.5

        fig_ts_arima.add_trace(go.Scatter(
            x=f_series.index.tolist() + f_series.index.tolist()[::-1],
            y=list(u_bound) + list(l_bound)[::-1],
            fill='toself',
            fillcolor='rgba(255,165,0,0.2)', 
            line=dict(color='rgba(255,255,255,0)'),
            name='Forecast Uncertainty'
        ))
        
        fig_ts_arima.update_layout(
            title="Patient Pulse Forecasting (30-day timeline, 7-day forecast)",
            xaxis_title="Date",
            yaxis_title="Pulse (BPM)",
            hovermode="x unified",
            template="plotly_dark", 
            margin=dict(l=0, r=0, b=0, t=40)
        )

        st.plotly_chart(fig_ts_arima, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error in ARIMA model: {e}")