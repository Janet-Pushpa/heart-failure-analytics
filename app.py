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


# --- REFINED DATA PIPELINE ---

@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    except:
        st.error("Dataset not found. Please ensure the CSV is in the project folder.")
        return None, None

    # A. Cleaning (Module 2)
    # We drop any duplicate rows that might skew the training
    df.drop_duplicates(inplace=True)
    
    # B. Feature Engineering (Domain Specific)
    # Clinical studies show the ratio of Serum Creatinine to Sodium is a strong indicator of kidney stress in heart patients.
    df['creatinine_sodium_ratio'] = df['serum_creatinine'] / df['serum_sodium']
    df['creatinine_age_ratio'] = df['serum_creatinine'] / df['age']

    # C. Feature Selection
    # We define numerical vs categorical for scaling logic
    numerical_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                      'platelets', 'serum_creatinine', 'serum_sodium', 'time',
                      'creatinine_sodium_ratio', 'creatinine_age_ratio']
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler

df, scaler = load_and_preprocess_data()

# --- MODEL TRAINING (The "Brain" Construction) ---

def train_risk_model(df):
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

    # 1. Stratified Split (Ensures equal ratio of Death/Life in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. SMOTE (Module 6: Rebalancing)
    # We only fit SMOTE on training data to prevent "Data Leakage"
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # 3. Model: Random Forest (Module 5: Classification)
    # n_estimators=100 creates an ensemble of 100 decision trees for voting.
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_bal, y_train_bal)
    
    return rf, X.columns, X_train_bal

rf_classifier, model_features, X_train_bal = train_risk_model(df)
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
# --- Tab 2: Diagnostics (Classification) ---
with tab2:
    st.header("🩺 Diagnostics: Heart Failure Risk")

    st.sidebar.header("Patient Input Features")

    # 1. Collect inputs
    age_input = st.sidebar.slider("Age", 40, 95, 60)
    cpk_input = st.sidebar.slider("Creatinine Phosphokinase", 20, 800, 200)
    ef_input = st.sidebar.slider("Ejection Fraction", 20, 80, 35)
    platelets_input = st.sidebar.slider("Platelets", 150000, 500000, 250000)
    sc_input = st.sidebar.slider("Serum Creatinine", 0.5, 9.0, 1.5)
    ss_input = st.sidebar.slider("Serum Sodium", 110, 150, 135)
    time_input = st.sidebar.slider("Follow-up Period", 4, 300, 150)

    anaemia_input = 1 if st.sidebar.checkbox("Anaemia") else 0
    diabetes_input = 1 if st.sidebar.checkbox("Diabetes") else 0
    hbp_input = 1 if st.sidebar.checkbox("High Blood Pressure") else 0
    sex_input = 1 if st.sidebar.radio("Sex", ["Male", "Female"]) == "Male" else 0
    smoking_input = 1 if st.sidebar.checkbox("Smoking") else 0

    # 2. Create the DataFrame with ALL features used during training
    input_dict = {
        'age': age_input,
        'anaemia': anaemia_input,
        'creatinine_phosphokinase': cpk_input,
        'diabetes': diabetes_input,
        'ejection_fraction': ef_input,
        'high_blood_pressure': hbp_input,
        'platelets': platelets_input,
        'serum_creatinine': sc_input,
        'serum_sodium': ss_input,
        'sex': sex_input,
        'smoking': smoking_input,
        'time': time_input
    }
    
    input_df = pd.DataFrame([input_dict])

    # 3. Add the engineered features (MUST match the training steps)
    input_df['creatinine_sodium_ratio'] = input_df['serum_creatinine'] / input_df['serum_sodium']
    input_df['creatinine_age_ratio'] = input_df['serum_creatinine'] / input_df['age']

    # 4. CRITICAL FIX: Reorder columns to match the model's training features exactly
    input_df = input_df[model_features] 

    # 5. Scale only the numerical columns in the EXACT order they were fit
    cols_to_scale = [
        'age', 'creatinine_phosphokinase', 'ejection_fraction', 
        'platelets', 'serum_creatinine', 'serum_sodium', 'time',
        'creatinine_sodium_ratio', 'creatinine_age_ratio'
    ]
    
    # Apply transformation
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    # 6. Prediction Button
    if st.sidebar.button("Predict Death Event Risk"):
        prediction_proba = rf_classifier.predict_proba(input_df)[:, 1][0]
        
        # UI Feedback
        color = "#00FF00" if prediction_proba < 0.3 else "#FFA500" if prediction_proba < 0.7 else "#FF0000"
        st.metric(label="Risk Probability", value=f"{prediction_proba * 100:.2f}%")
        
        style_metric_cards(background_color="#1f2630", border_left_color=color)

        # Gauge Chart (Note: use width='stretch' to fix warning)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction_proba * 100,
            title = {"text": "Risk Assessment"},
            gauge = {'axis': {'range': [0, 100]},
                     'steps': [
                         {'range': [0, 30], 'color': "green"},
                         {'range': [30, 70], 'color': "orange"},
                         {'range': [70, 100], 'color': "red"}]}
        ))
        st.plotly_chart(fig_gauge, width='stretch')

# --- FIXED SHAP LOGIC ---
        st.subheader("SHAP Waterfall Plot: Why this prediction?")
        
        # 1. Create the explainer
        explainer = shap.TreeExplainer(rf_classifier)
        shap_values = explainer.shap_values(input_df)

        # 2. Handle dimensionality (The fix for your IndexError)
        # Some versions return [samples, features, classes], others just [samples, features]
        if isinstance(shap_values, list):
            # If it's a list (older SHAP), take the values for the 'Death Event' (index 1)
            sv = shap_values[1][0]
            bv = explainer.expected_value[1]
        elif len(shap_values.shape) == 3:
            # If it's a 3D array, take [sample 0, all features, class 1]
            sv = shap_values[0, :, 1]
            bv = explainer.expected_value[1]
        else:
            # If it's already 2D, just take the first sample
            sv = shap_values[0]
            bv = explainer.expected_value

        # 3. Create the Explanation object for the waterfall plot
        exp = shap.Explanation(
            values=sv, 
            base_values=bv, 
            data=input_df.iloc[0].values, 
            feature_names=list(model_features)
        )

        # 4. Plotting
        fig_shap, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False)
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.clf()

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