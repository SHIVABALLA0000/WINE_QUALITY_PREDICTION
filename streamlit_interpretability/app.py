# streamlit_app/app.py

import streamlit as st

st.set_page_config(
    page_title="Wine Quality Prediction – ML System",
    layout="wide"
)

st.title("🍷 Wine Quality Prediction – ML System")

st.markdown("""
This application demonstrates a **production-grade machine learning system**
built with a **reliability-first mindset**.

### 🔹 What this system includes
- End-to-end ML pipeline (EDA → Modeling → Deployment)
- Stacked ensemble (Random Forest + XGBoost + Extra Trees)
- Optuna + Nested Cross-Validation
- Confidence-based prediction filtering
- Multi-level interpretability
- FastAPI inference backend
- Streamlit explainability dashboard

---

### 🔍 How to explore
Use the **left sidebar** to navigate through:
1. **Permutation Importance** – global feature relevance  
2. **PDP Sanity Checks** – behavioral validation  
3. **SHAP Explainability** – detailed feature attribution  

Each layer answers a *different trust question*.
""")

st.info("""
💡 Design principle:
Interpretability is layered — **sanity → behavior → explanation**.
SHAP is powerful, but it comes *after* simpler global checks.
""")
