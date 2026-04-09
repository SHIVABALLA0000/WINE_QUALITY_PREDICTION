#  Wine Quality Prediction  

A **production-grade machine learning system** designed with **reliability as the primary objective**, strengthened by **statistical validation, interpretability, monitoring, and safe inference practices**.

---

##  Project Overview  

This project builds an **end-to-end ML system** to predict **wine quality (ordinal multi-class classification)** using physicochemical features.

### Key Principles:
-  Reliable predictions  
-  Statistically validated model selection  
-  Explainability as a validation tool  
-  Deployment-ready inference  
-  Monitoring for post-deployment risks  

---

##  Problem Statement  

Predict **wine quality scores (ordinal multi-class)** while ensuring:

- Robust generalization  
- Fair performance across imbalanced classes  
- Statistical confidence in model comparisons  
- Confidence-aware predictions for real-world usage  

---

##  Dataset  

- **Source:** UCI Wine Quality Dataset  
- **Variants:** Red + White (combined)  
- **Target:** Ordinal quality score  

---

##  Modeling Approach  

### 🔹 Base Learners (Diversity-Driven)
- **Random Forest** – variance reduction  
- **XGBoost** – bias reduction  
- **Extra Trees** – decorrelation via randomness  

###  Meta-Learner
- **Logistic Regression**  
- Trained on base model probabilities  
- Produces calibrated and interpretable outputs  

###  Class Imbalance Handling
- Class-weighted loss  
- Macro-based evaluation  

---

##  Validation Strategy (No Data Leakage)  

- **Nested Cross-Validation**
  - Outer CV → unbiased performance estimate  
  - Inner CV → Optuna hyperparameter tuning  

- **Final Holdout Test Set**
  - Strictly untouched during training  

---

##  Evaluation Metrics  

- **Primary:** F1-macro (fairness across classes)  
- **Secondary:** RMSE (ordinal sensitivity)  

---

##  Statistical Model Evaluation  

To ensure **rigorous and defensible model selection**, we implemented **statistical testing and uncertainty estimation**:

###  1. Paired Model Comparison
- Paired t-test  
- Wilcoxon signed-rank test  
- Cohen’s d (effect size)  

 Validates whether performance differences are **statistically significant**, not just random.

---

###  2. Bootstrap Confidence Intervals
- Resampling-based estimation of metric distribution  

Provides:
- Mean F1-score  
- 95% Confidence Interval  

 Quantifies **uncertainty in model performance**

---

###  3. Calibration Evaluation
- Brier Score  
- Measures probability reliability  

 Ensures predicted probabilities are **trustworthy**

---

###  4. Statistical Reporting
- Results saved as JSON artifacts  
- Enables reproducibility & auditability  

---

##  Interpretability 

Performed before deployment:

- **Permutation Importance** → global feature sanity  
- **PDP (Partial Dependence Plots)** → domain validation  

### SHAP:
- Local + global explanations  
- Base-model inspection  
- Meta-learner interpretability  

---

##  Inference & Serving  

- **FastAPI backend**

Returns:
- Predicted class  
- Full probability distribution  
- Confidence score  

###  Safe Inference
- High → Accept  
- Medium → Warn  
- Low → Reject  

---

##  User Interface  

- **Streamlit frontend**

Features:
- Interactive feature input  
- Prediction visualization  
- Confidence score  
- Class probabilities  

---

##  Monitoring & Drift Detection  

- **Evidently AI**

Tracks:
- Feature drift  
- Target drift  

Compares:
- Training data  
- Simulated production data  

---

##  Key Engineering Highlight  

> Unlike typical ML projects, this system ensures that **model improvements are statistically validated, not assumed** — making it suitable for **high-stakes, production-grade deployments**.