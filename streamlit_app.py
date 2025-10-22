import streamlit as st
import joblib
import numpy as np
import os

# ==============================
# CONFIG
# ==============================
BASE_DIR = r"D:\Oil_Adulteration_Models"  # Root folder containing oil folders

st.set_page_config(page_title="Oil Adulteration Predictor", layout="centered")
st.title("🧪 Edible Oil Adulteration Prediction")
st.markdown("""
Predict the purity or quality of edible oils using pre-trained ML models.
Select an oil, choose a model, and enter the feature values manually.
""")

# ==============================
# Feature dictionary for each oil
# ==============================
feature_dict = {
    "Groundnut Oil": ["Reflectance", "Attenuation", "4pt Loss (db)", "A-B Loss (db)",
                      "A-B ORL (db)", "Total ORL", "IOR"],
    "Sunflower Oil": ["Reflectance", "Attenuation", "4 pt Loss", "A-B Loss",
                      "A-B ORL", "Total ORL", "Total loss(dB)", "IOR"]
}

# ==============================
# Select oil type (only folders with .joblib models)
# ==============================
oil_types = [
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d)) 
       and any(f.endswith(".joblib") for f in os.listdir(os.path.join(BASE_DIR, d)))
       and not d.startswith(".")
]

if not oil_types:
    st.error("⚠️ No oil folders with models found in the base directory.")
    st.stop()

selected_oil = st.selectbox("🛢️ Select Oil Type:", oil_types)

# ==============================
# List all loadable model files
# ==============================
model_dir = os.path.join(BASE_DIR, selected_oil)
all_model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]

# Attempt to pre-load each model and only include loadable ones
loadable_models = []
for f in all_model_files:
    try:
        joblib.load(os.path.join(model_dir, f))
        loadable_models.append(f)
    except Exception:
        continue

if not loadable_models:
    st.error(f"No loadable .joblib models found in folder: {model_dir}")
    st.stop()

selected_model_file = st.selectbox("🤖 Select Model:", loadable_models)
model_path = os.path.join(model_dir, selected_model_file)

# Load the selected model
model = joblib.load(model_path)
st.success(f"✅ Loaded model: `{selected_model_file}`")

# ==============================
# Input feature values (manual)
# ==============================
st.subheader("🔢 Enter Feature Values")
feature_names = feature_dict.get(selected_oil)

if not feature_names:
    st.error(f"No feature names defined for {selected_oil}")
    st.stop()

inputs = []
cols = st.columns(2)
for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        val_str = st.text_input(f"{feature}", value="0.0")  # User types manually
        try:
            val = float(val_str)
        except ValueError:
            val = 0.0
        inputs.append(val)

# ==============================
# Predict
# ==============================
if st.button("🔮 Predict"):
    try:
        X_input = np.array(inputs).reshape(1, -1)
        n_model_features = model.n_features_in_ if hasattr(model, "n_features_in_") else len(inputs)
        if X_input.shape[1] != n_model_features:
            st.error(f"❌ Number of inputs ({X_input.shape[1]}) does not match model requirement ({n_model_features})")
        else:
            prediction = model.predict(X_input)
            
            # Clip prediction to range 2-20
            prediction_clipped = np.clip(prediction, 2, 20)
            
            st.success(f"**Predicted {selected_oil} value:** {prediction_clipped[0]:.3f}")
            st.balloons()
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.caption("Developed by Kabilesh Raj — Powered by Streamlit & scikit-learn")
