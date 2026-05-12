import os
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# 1. App paths and cached loading
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR


@st.cache_resource
def load_model_artifacts():
    pipeline_path = WEB_DIR / "final_pipeline.pkl"
    meta_path = WEB_DIR / "model_meta.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Missing model file: {pipeline_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    with open(pipeline_path, "rb") as f:
        final_pipeline = pickle.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return final_pipeline, meta


@st.cache_data
def load_shap_background():
    bg_path = WEB_DIR / "shap_background.csv"

    if not bg_path.exists():
        raise FileNotFoundError(f"Missing SHAP background file: {bg_path}")

    return pd.read_csv(bg_path)


final_pipeline, meta = load_model_artifacts()
X_bg = load_shap_background()

candidate_features = meta.get("candidate_features", [])
optimal_threshold = float(meta["optimal_threshold"])
best_model_name = meta["best_model_name"]


# =========================================================
# 2. Page settings
# =========================================================
st.set_page_config(
    page_title="Psoriasis-Associated Insulin Resistance Risk Prediction Tool",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
.main > div {
    max-width: 1100px;
    padding-top: 1.5rem;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
.stButton > button {
    background-color: #d9534f;
    color: white;
    border: 2px solid #d9534f;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
}
.stButton > button:hover {
    background-color: white;
    color: #d9534f;
    border: 2px solid #d9534f;
}
.result-note {
    text-align: center;
    font-size: 1.15rem;
    font-style: italic;
    margin-top: 0.8rem;
    margin-bottom: 1rem;
}
.small-note {
    color: #666;
    font-size: 0.95rem;
}
.disclaimer {
    background-color: #fff3cd;
    border-left: 5px solid #ffca2c;
    padding: 0.8rem 1rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center; font-size: 2.2rem;'>"
    "Psoriasis-Associated Insulin Resistance Risk Prediction Tool"
    "</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;' class='small-note'>"
    "Enter the required clinical variables to obtain an individualized estimate of TyG-defined insulin resistance risk."
    "</p>",
    unsafe_allow_html=True
)


# =========================================================
# 3. Input form
# =========================================================
with st.form("prediction_form", clear_on_submit=False):
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
    sex_label = st.selectbox("Sex", ["Female", "Male"])
    sex = 0 if sex_label == "Female" else 1

    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)
    pasi = st.number_input("PASI score", min_value=0.0, max_value=72.0, value=8.0, step=0.1)
    duration = st.number_input("Disease duration (years)", min_value=0.0, max_value=80.0, value=10.0, step=0.1)

    hdl = st.number_input("HDL-C", min_value=0.1, max_value=5.0, value=1.2, step=0.01)
    ldl = st.number_input("LDL-C", min_value=0.1, max_value=10.0, value=2.8, step=0.01)

    submitted = st.form_submit_button("Predict")


# =========================================================
# 4. Prediction
# =========================================================
if submitted:
    input_df = pd.DataFrame([{
        "age": float(age),
        "sex": int(sex),
        "bmi": float(bmi),
        "pasi": float(pasi),
        "duration": float(duration),
        "hdl": float(hdl),
        "ldl": float(ldl),
    }])

    prob = float(final_pipeline.predict_proba(input_df)[:, 1][0])
    pred_class = int(prob >= optimal_threshold)

    risk_pct = prob * 100
    pred_text = "IR" if pred_class == 1 else "Non-IR"

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Prediction Result</h2>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='result-note'>"
        f"Predicted probability of TyG-defined insulin resistance: <b>{risk_pct:.1f}%</b>."
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div style='text-align:center; font-size:1.1rem; margin-bottom:0.8rem;'>"
        f"<b>Predicted class:</b> {pred_text}"
        f"</div>",
        unsafe_allow_html=True
    )

    with st.expander("Show technical details"):
        st.write(f"**Final model:** {best_model_name}")
        st.write(f"**Predicted probability of IR:** {prob:.3f}")
        st.write(f"**Locked threshold:** {optimal_threshold:.3f}")
        st.write("**Input variables:** age, sex, BMI, PASI score, disease duration, HDL-C, LDL-C")

    # =====================================================
    # 5. SHAP force plot
    # =====================================================
    preprocess = final_pipeline.named_steps["preprocess"]
    model = final_pipeline.named_steps["model"]

    X_bg_t = preprocess.transform(X_bg)
    X_input_t = preprocess.transform(input_df)

    if hasattr(X_bg_t, "toarray"):
        X_bg_t = X_bg_t.toarray()
    if hasattr(X_input_t, "toarray"):
        X_input_t = X_input_t.toarray()

    X_bg_t = np.asarray(X_bg_t)
    X_input_t = np.asarray(X_input_t)

    try:
        feature_names = preprocess.get_feature_names_out().tolist()
        feature_names = [str(x).replace("num__", "").replace("cat__", "") for x in feature_names]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_input_t.shape[1])]

    tree_model_names = [
        "RandomForestClassifier", "ExtraTreesClassifier",
        "GradientBoostingClassifier", "XGBClassifier",
        "LGBMClassifier", "CatBoostClassifier",
        "DecisionTreeClassifier"
    ]

    model_name = model.__class__.__name__

    if model_name in tree_model_names or hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input_t)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        base_value = explainer.expected_value

        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if np.ndim(base_value) > 0 else float(base_value)

        base_value = float(np.asarray(base_value))

    else:
        bg_small = X_bg_t[:min(50, len(X_bg_t))]

        def pred_fn(x):
            return model.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(pred_fn, bg_small)
        shap_values = explainer.shap_values(X_input_t, nsamples="auto")

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        base_value = explainer.expected_value

        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if np.ndim(base_value) > 0 else float(base_value)

        base_value = float(np.asarray(base_value))


    force = shap.force_plot(
        base_value=base_value,
        shap_values=shap_values[0],
        features=X_input_t[0],
        feature_names=feature_names,
        matplotlib=False
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as tmp:
        shap.save_html(tmp.name, force)
        temp_html_path = tmp.name

    with open(temp_html_path, "r", encoding="utf-8") as f:
        html_string = f.read()

    try:
        os.remove(temp_html_path)
    except OSError:
        pass

    st.markdown(
        "<h3 style='text-align:center; margin-top:1rem;'>Individualized SHAP Force Plot</h3>",
        unsafe_allow_html=True
    )

    components.html(html_string, height=280, scrolling=False)

    # =====================================================
    # 6. Input summary
    # =====================================================
    st.subheader("Input Summary")
    st.dataframe(input_df, use_container_width=True)
