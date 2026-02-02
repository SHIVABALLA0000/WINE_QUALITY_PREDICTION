import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(layout="wide")

st.title("🔁 Permutation Importance (Global Trust)")
st.markdown("""
**Permutation Importance** measures how much model performance drops when a
feature is randomly shuffled.

✔ Model-agnostic  
✔ Works with stacking  
✔ First sanity check before SHAP
""")

# -----------------------------
# Load precomputed importance
# -----------------------------
PATH = Path("wine_artifacts/permutation_importance.csv")

if not PATH.exists():
    st.error("Permutation importance file not found. Run permutation_importance.py first.")
    st.stop()

perm_df = pd.read_csv(PATH)

# -----------------------------
# Display
# -----------------------------
st.subheader("Top Influential Features")

top_k = st.slider("Show top K features", 5, 30, 15)
st.dataframe(
    perm_df.head(top_k),
    use_container_width=True
)

st.bar_chart(
    perm_df.head(top_k).set_index("feature")["importance_mean"]
)

st.markdown("""
### 🧠 How to interpret
- Higher importance → feature strongly affects predictions
- Near-zero importance → feature has little effect
- If random/noise features appear high → ⚠️ investigate leakage
""")
