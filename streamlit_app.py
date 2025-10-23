import streamlit as st
import joblib
import numpy as np
import os

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.getcwd()
st.set_page_config(page_title="Oil Adulteration Predictor", layout="centered")

st.title("🧪 Edible Oil Adulteration Prediction")
st.markdown("""
Predict the purity or quality of edible oils using pre-trained ML models.  
Select an oil, choose a model, and enter the feature values manually.
""")

# ==============================
# FEATURE DICTIONARY
# ==============================
feature_dict = {
    "Groundnut Oil": [
        "Reflectance", "Attenuation", "4pt Loss (db)", "A-B Loss (db)",
        "A-B ORL (db)", "Total ORL", "IOR"
    ],
    "Sunflower Oil": [
        "Reflectance", "Attenuation", "4 pt Loss", "A-B Loss",
        "A-B ORL", "Total ORL", "Total loss(dB)", "IOR"
    ],
    "Gingelly Oil": [
        "Reflectance", "Attenuation", "ORL", "IOR"
    ],
    "Mustard Oil": [
        "Reflectance", "Attenuation", "4pt Loss (db)", "A-B Loss (db)",
        "A-B ORL (db)", "Total ORL (db)", "IOR"
    ]
}

# ==============================
# DETECT OIL FOLDERS
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
# DETECT AND LOAD MODELS
# ==============================
model_dir = os.path.join(BASE_DIR, selected_oil)
all_model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]

loadable_models = []
failed_models = []
for f in all_model_files:
    try:
        joblib.load(os.path.join(model_dir, f))
        loadable_models.append(f)
    except Exception as e:
        failed_models.append((f, str(e)))

if not loadable_models:
    st.error(f"No valid .joblib models found for {selected_oil}.")
    st.stop()

# Create pretty display names
model_display_names = [os.path.splitext(f)[0].replace("_", " ") for f in loadable_models]
selected_display_name = st.selectbox("🤖 Select Model:", model_display_names)

# Map back to file
selected_model_file = loadable_models[model_display_names.index(selected_display_name)]
model_path = os.path.join(model_dir, selected_model_file)
model = joblib.load(model_path)

st.success(f"✅ Loaded model: {selected_display_name}")

# If any models failed to load, show in debug mode
if failed_models:
    with st.expander("⚠️ Some models could not be loaded (click for details)"):
        for f, err in failed_models:
            st.write(f"❌ **{f}** — {err}")

# ==============================
# FEATURE INPUT SECTION
# ==============================
st.subheader("🔢 Enter Feature Values")

feature_names = feature_dict.get(selected_oil)
if not feature_names:
    st.warning(f"No predefined features for {selected_oil}. Using default generic inputs.")
    feature_names = [f"Feature {i+1}" for i in range(7)]

inputs = []
cols = st.columns(2)
for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        val_str = st.text_input(f"{feature}", value="0.0")
        try:
            val = float(val_str)
        except ValueError:
            val = 0.0
        inputs.append(val)

# ==============================
# PREDICT BUTTON
# ==============================
if st.button("🔮 Predict"):
    try:
        X_input = np.array(inputs).reshape(1, -1)
        n_model_features = getattr(model, "n_features_in_", len(inputs))
        if X_input.shape[1] != n_model_features:
            st.error(f"❌ Number of inputs ({X_input.shape[1]}) does not match model requirement ({n_model_features}).")
        else:
            prediction = model.predict(X_input)
            prediction_clipped = np.clip(prediction, 2, 20)
            st.success(f"**Predicted {selected_oil} value:** {prediction_clipped[0]:.3f}")
            st.balloons()
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("Developed by Kabilesh Raj — Powered by Streamlit & scikit-learn")
