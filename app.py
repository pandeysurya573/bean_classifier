import streamlit as st
import numpy as np
import joblib

# Page setup
st.set_page_config(page_title="Dry Bean Classifier", page_icon="🌱", layout="centered")

st.title("🌱 Dry Bean Type Classifier")
st.write("Enter physical and morphological measurements captured by the optical imaging system to classify bean variety.")

# Load models and preprocessing pipelines
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_svm_bean_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return model, scaler, encoder

try:
    model, scaler, encoder = load_artifacts()
except Exception as e:
    st.error("Model artifacts not found! Please run the training script first to generate .pkl files.")
    st.stop()

# Input layout in 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("Area (A)", min_value=10000.0, max_value=300000.0, value=42000.0, step=500.0)
    perimeter = st.number_input("Perimeter (P)", min_value=300.0, max_value=2500.0, value=760.0, step=10.0)
    major_axis = st.number_input("Major Axis Length (L)", min_value=100.0, max_value=800.0, value=280.0, step=5.0)
    minor_axis = st.number_input("Minor Axis Length (l)", min_value=50.0, max_value=600.0, value=190.0, step=5.0)
    aspect_ratio = st.number_input("Aspect Ratio (K)", min_value=1.0, max_value=3.5, value=1.47, step=0.01)
    eccentricity = st.number_input("Eccentricity (Ec)", min_value=0.1, max_value=1.0, value=0.73, step=0.01)

with col2:
    convex_area = st.number_input("Convex Area (C)", min_value=10000.0, max_value=300000.0, value=42600.0, step=500.0)
    equiv_diameter = st.number_input("Equivalent Diameter (Ed)", min_value=100.0, max_value=700.0, value=231.0, step=5.0)
    extent = st.number_input("Extent (Ex)", min_value=0.4, max_value=0.9, value=0.75, step=0.01)
    solidity = st.number_input("Solidity (S)", min_value=0.8, max_value=1.0, value=0.98, step=0.005)
    roundness = st.number_input("Roundness (R)", min_value=0.4, max_value=1.0, value=0.88, step=0.01)

with col3:
    compactness = st.number_input("Compactness (CO)", min_value=0.4, max_value=1.0, value=0.82, step=0.01)
    sf1 = st.number_input("Shape Factor 1 (SF1)", min_value=0.001, max_value=0.02, value=0.0066, format="%.5f")
    sf2 = st.number_input("Shape Factor 2 (SF2)", min_value=0.0005, max_value=0.01, value=0.0022, format="%.5f")
    sf3 = st.number_input("Shape Factor 3 (SF3)", min_value=0.2, max_value=1.0, value=0.68, step=0.01)
    sf4 = st.number_input("Shape Factor 4 (SF4)", min_value=0.8, max_value=1.0, value=0.99, step=0.001)

st.markdown("---")

if st.button("Classify Bean", type="primary", use_container_width=True):
    input_features = np.array([[
        area, perimeter, major_axis, minor_axis, aspect_ratio, eccentricity,
        convex_area, equiv_diameter, extent, solidity, roundness, compactness,
        sf1, sf2, sf3, sf4
    ]])
    
    # Scale inputs using fitted scaler
    input_scaled = scaler.transform(input_features)
    
    # Inference
    prediction_idx = model.predict(input_scaled)[0]
    predicted_bean = encoder.inverse_transform([prediction_idx])[0]
    
    st.success(f"### Predicted Bean Variety: **{predicted_bean}**")