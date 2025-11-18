import streamlit as st
import joblib
import numpy as np
import os

# --- CONFIG ---
BASE_DIR = os.getcwd()
st.set_page_config(page_title="Oil Adulteration Predictor", layout="centered")

st.title("Edible Oil Adulteration Prediction")

# --- FEATURE SETS ---
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
        "Reflectance", "Attenuation", "4pt Loss (db)", "A-B Loss (db)",
        "A-B ORL (db)", "Total ORL (db)", "IOR"
    ],
    "Mustard Oil": [
        "Reflectance", "Attenuation", "4pt Loss (db)", "A-B Loss (db)",
        "A-B ORL (db)", "Total ORL (db)", "IOR"
    ]
}

# --- DETECT OIL FOLDERS ---
oil_types = [
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
    and any(f.endswith(".joblib") for f in os.listdir(os.path.join(BASE_DIR, d)))
]

if not oil_types:
    st.error("No oil folders with models found.")
    st.stop()

selected_oil = st.selectbox("Select Oil Type:", sorted(oil_types))

# --- LOAD MODELS ---
model_dir = os.path.join(BASE_DIR, selected_oil)
model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]

loadable = []
for f in model_files:
    try:
        joblib.load(os.path.join(model_dir, f))
        loadable.append(f)
    except:
        pass

if not loadable:
    st.error("No valid joblib models found.")
    st.stop()

model_names = [os.path.splitext(f)[0].replace("_", " ").title() for f in loadable]
selected_name = st.selectbox("Select Model:", model_names)

model_file = loadable[model_names.index(selected_name)]
model = joblib.load(os.path.join(model_dir, model_file))

# --- INPUT SECTION ---
st.subheader("Enter Feature Values")
feature_names = feature_dict.get(selected_oil, [])

raw_inputs = {}
cols = st.columns(2)

for i, feature in enumerate(feature_names):
    if feature.upper().startswith("IOR"):
        continue
    with cols[i % 2]:
        raw_inputs[feature] = st.text_input(feature, "", placeholder="Enter value")

# --- IOR CALC ---
def compute_ior(R):
    try:
        return 1.4683 * ((1 - 10 ** (R / 20)) / (1 + 10 ** (R / 20)))
    except:
        return None

ior_displayed = None
if "Reflectance" in raw_inputs and raw_inputs["Reflectance"].strip() != "":
    try:
        r_val = float(raw_inputs["Reflectance"])
        ior_displayed = compute_ior(r_val)
        if ior_displayed is not None:
            st.info(f"IOR = {ior_displayed:.4f}")
        else:
            st.warning("Invalid Reflectance for IOR calculation.")
    except:
        st.warning("Reflectance must be numeric to compute IOR.")

# --- PREDICT ---
if st.button("Predict"):
    errors = []
    numeric_inputs = {}

    for feature, val in raw_inputs.items():
        if val.strip() == "":
            errors.append(f"{feature} empty")
            continue
        try:
            numeric_inputs[feature] = float(val)
        except:
            errors.append(f"{feature} not numeric")

    if "Reflectance" in numeric_inputs:
        ior_val = compute_ior(numeric_inputs["Reflectance"])
        if ior_val is None:
            errors.append("Unable to compute IOR from Reflectance")
    else:
        errors.append("Reflectance required for IOR")

    if errors:
        st.error("Fix the following issues:")
        for e in errors:
            st.write("- " + e)
        st.stop()

    numeric_inputs["IOR"] = ior_val

    ordered = feature_dict[selected_oil]
    final_vals = [numeric_inputs.get(f, 0.0) for f in ordered]
    X = np.array(final_vals).reshape(1, -1)

    expected = getattr(model, "n_features_in_", X.shape[1])
    if X.shape[1] != expected:
        st.error(f"Feature mismatch: model expects {expected}, got {X.shape[1]}")
        st.stop()

    try:
        pred = model.predict(X)[0]
        pred = np.clip(pred, 0, 20)   # UPDATED RANGE

        # --- NEW CALCULATIONS ---
        palm_oil_ml = 20 - pred
        adulteration_pct = (palm_oil_ml / 20) * 100

        st.success(f"Predicted {selected_oil} Value: {pred:.3f}")
        st.info(f"Palm Oil : {palm_oil_ml:.3f} ml")
        st.warning(f"Adulteration Percentage : {adulteration_pct:.2f}%")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Developed by T Kabilesh Raj — Powered by Streamlit")
