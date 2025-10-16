# Goal: Let users interact with the CharityML model by inputting
#       their data and getting real-time donation probability.
# ============================================================

# ========================
# ========================
# 1️⃣ Import Dependencies
# ========================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from PIL import Image

# 🚨 التعديل الضروري لإصلاح الخطأ: إضافة استيراد PyCaret
# هذا يحل مشكلة ModuleNotFoundError عند تحميل ملف .joblib
import pycaret 

# (قم بإزالة استيرادات FLAML و AutoML إن لم تكن تستخدمها بالفعل في أماكن أخرى من الكود)
# import flaml 
# from flaml import AutoML 
# ...

# ========================
# 2️⃣ Page Configuration
# ========================
st.set_page_config(
    page_title="🎯 CharityML Donation Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Header Logo
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#2E86C1;">💡 CharityML - Smart Donation Predictor</h1>
        <h4 style="color:#A93226;">Powered by Shahinda </h4>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ========================
# 3️⃣ Load Model & Encoders
# ========================
@st.cache_resource
def load_assets():
    # 📝 التعديل الثاني: استخدام مسار نسبي لتحميل الموديل
    # بما أن main.py موجود في Finding_Donors، نستخدم اسم الملف مباشرة.
    model = joblib.load("final_automl_best_model.joblib")      # Final stacked model
    
    # 📝 التعديل الثالث: استخدام مسار نسبي لتحميل الـ encoder والـ scaler
    # يجب التأكد من أن هذه الملفات موجودة في نفس مجلد main.py
    encoder = joblib.load("encoder.joblib")             # OneHot/Label encoder
    scaler = joblib.load("scaler.joblib")               # Scaler
    return model, encoder, scaler

# يتم استدعاء الدالة وتحميل الأصول
model, encoder, scaler = load_assets()

# ========================
# 4️⃣ User Input Section
# ========================
st.markdown("### 🧠 Enter Person Details")

with st.form("donor_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 90, 35)
        education = st.selectbox("Education Level", [
            "Preschool", "HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate"
        ])
        marital_status = st.selectbox("Marital Status", [
            "Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed"
        ])

    with col2:
        occupation = st.selectbox("Occupation", [
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
            "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving"
        ])
        hours_per_week = st.slider("Hours per Week", 1, 100, 40)
        country = st.selectbox("Country", ["United-States", "Canada", "Mexico", "India", "Egypt", "Germany"])

    submitted = st.form_submit_button("🔮 Predict Donation Probability")

# ========================
# 5️⃣ Data Preprocessing
# ========================
def preprocess_input():
    input_df = pd.DataFrame({
        "age": [age],
        "education": [education],
        "occupation": [occupation],
        "hours-per-week": [hours_per_week],
        "marital-status": [marital_status],
        "native-country": [country]
    })

    # Apply same preprocessing as training
    numeric_features = ["age", "hours-per-week"]
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    # One-hot encode categorical variables using pre-trained encoder
    # ملاحظة: تم تعديل السطر التالي للتعامل مع DataFrame
    # حيث أن encoder.transform() في الـ scikit-learn الحديثة بترجع مصفوفة (Array)
    encoded_array = encoder.transform(input_df)
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())

    return encoded_df

# ========================
# 6️⃣ Prediction & SDI Calculation
# ========================
if submitted:
    st.markdown("---")
    st.markdown("### 🔍 Predicting... Please wait a second ⏳")

    try:
        X_input = preprocess_input()
        # يتم استخدام .iloc[0] للتأكد من أن الإدخال صف واحد إذا كان هناك مشكلة في الـ reshape
        prediction_prob = model.predict_proba(X_input.values)[0][1] * 100

        # Smart Donation Index formula (for transparency)
        # تم تعديل طريقة حساب الـ SDI لتفادي أي أخطاء محتملة
        
        # 1. Age factor
        age_factor = age / 90 
        
        # 2. Education factor
        if education in ["Masters", "Doctorate"]:
            education_factor = 1.0
        elif education in ["Bachelors"]:
            education_factor = 0.8
        else:
            education_factor = 0.6
            
        # 3. Hours factor
        hours_factor = hours_per_week / 100
        
        # 4. Marital factor
        marital_factor = 1.0 if marital_status.startswith("Married") else 0.5
        
        sdi = round(
            (0.4 * education_factor) +
            (0.3 * hours_factor) +
            (0.2 * age_factor) +
            (0.1 * marital_factor),
            3
        )


        # ========================
        # 🎯 Output Results
        # ========================
        st.success(f"💰 Donation Probability (FLAML Model): *{prediction_prob:.2f}%*")
        st.info(f"🤖 Smart Donation Index (SDI++): *{sdi}*")

        if prediction_prob >= 75:
            st.markdown("### 🔥 ممتاز! فرصة تبرع عالية جداً. (High Chance of Donation)")
        elif prediction_prob >= 50:
            st.markdown("### ⚡ فرصة تبرع متوسطة. (Moderate Chance of Donation)")
        else:
            st.markdown("### 💤 فرصة تبرع منخفضة. (Low Donation Likelihood)")

        # ========================
        # 🧩 SHAP Explanation (Explainable AI)
        # ========================
        with st.expander("🔍 Explanations (SHAP Insights)"):
            try:
                # نحتاج إلى تمرير الـ X_input (DataFrame) للـ explainer في بعض الحالات
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_input)
                
                # SHAP plots for multi-class models usually need shap_values[1] for the positive class
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values_pos_class = shap_values[1]
                else:
                    shap_values_pos_class = shap_values
                    
                # إنشاء Shap values object لتمريره للـ waterfall plot
                shap_object = shap.Explanation(
                    values=shap_values_pos_class[0], 
                    base_values=explainer.expected_value, 
                    data=X_input.values[0], 
                    feature_names=X_input.columns.tolist()
                )
                
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(shap.plots.waterfall(shap_object, show=False))
            except Exception as e:
                # قد يكون الخطأ هو TypeError: cannot compute Shap values for this model
                st.warning(f"⚠ SHAP visualization غير متوفرة (Error: {e}).")

    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء المعالجة أو التنبؤ: {e}")

# ========================
# 7️⃣ Footer
# ========================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🧠 <b>CharityML Project</b> | Developed by <b>Shahinda</b> 🌟 <br>

    </div>
    """,
    unsafe_allow_html=True
)