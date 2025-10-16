
# 🧠 CharityML Model Card

---

## 🏷 Model Overview

*Model Name:* CharityML Smart Donor Classifier  
*Version:* v3.2   
*Author:* Shahinda 🌟  
*Date:* October 2025  
*Frameworks:* Scikit-learn | XGBoost | LightGBM | PyCaret  
*Goal:* Predict whether an individual is likely to donate to charity based on census data.

---

## 📊 Model Details

| Category | Description |
|-----------|-------------|
| *Algorithm Type* | Ensemble Stacking (XGBoost + LightGBM + Logistic Regression) |
| *Problem Type* | Binary Classification |
| *Target Variable* | donate → (1 if income > 50K, else 0) |
| *Input Data* | U.S. Census Income Dataset |
| *Training Data Size* | 39,000 records |
| *Test Data Size* | 9,700 records |
| *Input Features* | Age, Education, Occupation, Hours-per-week, Marital-status, etc. |
| *Output* | Probability of donation likelihood (0–1) |

---

## 🎯 Intended Use

This model is designed to help *non-profit organizations and charities*:
- Identify potential donors.
- Optimize marketing campaigns by targeting likely contributors.
- Increase donation conversion rates through data-driven insights.

*✅ Suitable Use Cases:*
- Predictive analytics for charity donation likelihood.  
- Integrating into CRM systems for donor segmentation.  
- Research on socio-economic donation patterns.

---

## ⚠ Limitations & Ethical Considerations

| Aspect | Description |
|---------|-------------|
| *Data Bias* | Based on U.S. census data — may not generalize globally. |
| *Socioeconomic Sensitivity* | Features like income or education may reflect biases if used improperly. |
| *Fairness* | Avoid using results for discrimination, employment, or credit decisions. |
| *Privacy* | Ensure data anonymization before inference. |

🧩 The model predicts donation potential — not generosity, moral behavior, or actual donation.  
It should *complement* human judgment, not replace it.

---

## 📈 Performance Summary

| Metric | Score |
|---------|-------|
| *Accuracy* | 0.93 |
| *Precision* | 0.91 |
| *Recall* | 0.88 |
| *F1-score* | 0.89 |
| *AUC* | 0.94 |

*Validation Strategy:* 80/20 split with Stratified Sampling  
*Cross-validation:* 5-fold cross-validation (average metrics reported)

---

## 💡 Key Features Impacting Prediction

| Rank | Feature | Influence |
|------|----------|-----------|
| 1️⃣ | Education Level | +++ |
| 2️⃣ | Hours-per-Week | ++ |
| 3️⃣ | Age | ++ |
| 4️⃣ | Marital Status | + |
| 5️⃣ | Occupation | + |

(Computed using SHAP Explainability Framework)

---

## 🧮 Smart Donation Index (SDI++)

*Formula:*  
SDI = (education_level * 0.4) + (hours_per_week * 0.3) + (age_factor * 0.2) + (marital_status_factor * 0.1)

*Interpretation:*  
Higher SDI values → higher likelihood of being an active donor.  
Average SDI (test data): *0.67 / 1.00*

---

## ☁ Deployment Details

| Component | Platform |
|------------|-----------|
| *Web App* | Streamlit Cloud (Interactive Dashboard) |
| *API* | FastAPI |
| *Model Storage* | joblib (.pkl) file |
| *Auto Report* | PDF Generator (ReportLab + FPDF) |

---

## 🧩 Maintenance & Updates

| Version | Changes |
|----------|----------|
| *v1.0* | Baseline models trained (Logistic, DT, RF) |
| *v2.0* | Feature engineering, scaling, and ensemble stacking added |
| *v3.0* | AutoML + SHAP Explainability + PDF Reporting integrated |
| *v3.2* | API + Cloud deployment finalized for Hackathon submission |

---

## 🧭 Next Steps

- Expand dataset to multi-regional donors.  
- Integrate real-time donation tracking APIs.  
- Add A/B testing for model calibration.  
- Introduce Federated Learning for privacy-preserving training.

---

## 🏆 Summary

> “CharityML Smart Donor Classifier” combines explainable AI, AutoML, and ensemble learning  
> to predict donor behavior with exceptional accuracy and transparency.  
> Developed by *Shahinda*, this model demonstrates the fusion of technical excellence  
> and real-world applicability — a true data-driven innovation in charity analytics.

---

*Created by:* Shahinda 🌟 — Data Scientist & ML Engineer  
📧 Contact: [LinkedIn Profile](https://www.linkedin.com/in/shahinda-ibrahim)  
📅 Last Updated: October 2025