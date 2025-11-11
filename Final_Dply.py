#!/usr/bin/env python
# coding: utf-8

# In[1]:


#===================Deployment (Streamlit App)===================================
#Steps 1
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Steps-2
#==========================Load Pretrained Model ==================================
try:
    model = pickle.load(open("Bankruptcy_Model_Final.pkl", "rb"))
except:
    model = None
#Steps-3
# ==================================Page Config ===================================
st.set_page_config(page_title="Bankruptcy Prediction App", page_icon="💰", layout="wide")
#Steps-4
#============================== Sidebar Navigation ================================
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to:", ["Home", "Predict", "Model Building", "Project Info", "About Developer"]
)
#Steps-5
#=====================================HOME PAGE ===================================
if menu == "Home":
    # Title with subtitle
    st.markdown(
        """
        <div style="text-align:center; padding: 10px;">
            <h1 style="color:#0073e6;">🏦 Bankruptcy Prediction System</h1>
                                      AI-powered tool to assess financial risk and predict potential bankruptcy.
        </div>
        """, unsafe_allow_html=True
    )

    # Add hero image (modern finance theme)
    st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://cdn-icons-png.flaticon.com/512/2721/2721298.png" width="350">
    </div>
    """,
    unsafe_allow_html=True
)

    # Horizontal line
    st.markdown("---")

    # Objective section
    st.markdown(
    """
    <div style="text-align: center; margin-top: 20px;">
        <h3>🎯 <b>Objective</b></h3>
        <p style="font-size:18px;">
        Predict whether a company might <b>go bankrupt</b> based on its financial risk factors.  
        This tool provides <b>early warnings</b> to help make better business decisions and prevent losses.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
    # Key Features
    st.markdown("### ⚙️ **Key Features**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🤖 Machine Learning Models")
        st.write("""
        - Logistic Regression  
        - Decision Tree  
        - Random Forest  
        - SVM  
        - KNN  
        - Gradient Boosting  
        - XGBoost  
        """)

    with col2:
        st.markdown("#### 📈 Data Insights")
        st.write("""
        - Real-time prediction dashboard  
        - Visual risk representation  
        - Comparative model performance  
        - EDA & statistical summaries  
        """)

    with col3:
        st.markdown("#### 🧠 Smart Prediction")
        st.write("""
        - Input flexibility (0, 0.5, 1)  
        - Instant results  
        - Accurate risk probability  
        - Downloadable reports *(coming soon)*  
        """)

    st.markdown("---")

    # About dataset summary
    st.markdown(
        """
        ### 🗂️ **Dataset Overview**
        The dataset contains financial risk indicators for **250 companies** with **7 key features:**
        - Industrial Risk  
        - Management Risk  
        - Financial Flexibility  
        - Credibility  
        - Competitiveness  
        - Operating Risk  
        - Class (Bankruptcy / Non-Bankruptcy)
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Call-to-action button
    st.markdown(
        """
        <div style="text-align:center; margin-top:20px;">
            <a href="#" target="_self">
                <button style="background-color:#0073e6; color:white; padding:12px 30px; 
                border:none; border-radius:10px; font-size:18px; cursor:pointer;">
                🚀 Get Started with Prediction
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True
    )

#Steps-6
# ================================PREDICTION PAGE ===============================
elif menu == "Predict":
    st.title("🤖 Bankruptcy Risk Prediction")
    st.markdown("Use your trained model to predict whether a company is at risk of bankruptcy based on input features.")
    st.markdown("---")

    # Load saved model safely
    try:
        model = pickle.load(open("Bankruptcy_Model_Final.pkl", "rb"))
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
        st.error("❌ Model not found! Please train a model first in the 'Model Building' section.")

    if model_loaded:
        st.markdown("### 📊 Enter Company Financial Indicators")
        st.write("*(Set each slider between 0 - 1, where 0=Low, 0.5=Medium, 1=High)*")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                industrial_risk = st.slider("🏭 Industrial Risk", 0.0, 1.0, step=0.5)
                financial_flexibility = st.slider("💰 Financial Flexibility", 0.0, 1.0, step=0.5)
                competitiveness = st.slider("📈 Competitiveness", 0.0, 1.0, step=0.5)

            with col2:
                management_risk = st.slider("👔 Management Risk", 0.0, 1.0, step=0.5)
                credibility = st.slider("🤝 Credibility", 0.0, 1.0, step=0.5)
                operating_risk = st.slider("⚙️ Operating Risk", 0.0, 1.0, step=0.5)

            st.markdown(" ")
            submit_button = st.form_submit_button("🔍 Predict Risk")

        if submit_button:
            # Prepare input
            input_data = np.array([[industrial_risk, management_risk,
                                    financial_flexibility, credibility,
                                    competitiveness, operating_risk]])

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] * 100

            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            if prediction == 1:
                st.error(f"⚠️ **High Bankruptcy Risk Detected!**  \nEstimated probability: **{probability:.2f}%**")
            else:
                st.success(f"✅ **Low Bankruptcy Risk Detected!**  \nEstimated probability: **{probability:.2f}%**")

            # Pie Chart Visualization
            st.markdown("### 📉 Risk Probability Chart")
            labels = ["Non-Bankruptcy", "Bankruptcy"]
            values = [100 - probability, probability]

            fig, ax = plt.subplots(figsize=(4, 4))
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=["#66b3ff", "#ff6666"],
                textprops={"color": "black", "fontsize": 10},
            )
            plt.setp(autotexts, size=11, weight="bold")

            # Save pie chart to buffer for PDF export ✅
            from io import BytesIO
            pie_buffer = BytesIO()
            plt.savefig(pie_buffer, format="png")
            pie_buffer.seek(0)
            st.pyplot(fig)

            # Display input summary
            st.markdown("### 📋 Entered Input Summary")
            input_df = pd.DataFrame(input_data, columns=[
                "Industrial Risk", "Management Risk", "Financial Flexibility",
                "Credibility", "Competitiveness", "Operating Risk"
            ])
            st.dataframe(input_df.style.highlight_max(axis=1, color="lightblue"))

            # ----- PDF GENERATION -----
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from datetime import datetime
            from reportlab.lib.utils import ImageReader

            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=A4)
            width, height = A4

            # Title
            c.setFont("Helvetica-Bold", 18)
            c.drawCentredString(width / 2, height - 50, "Bankruptcy Prediction Report")

            # Timestamp
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Inputs
            y = height - 110
            c.setFont("Helvetica", 12)
            c.drawString(50, y, f"Industrial Risk: {industrial_risk}")
            c.drawString(50, y - 20, f"Management Risk: {management_risk}")
            c.drawString(50, y - 40, f"Financial Flexibility: {financial_flexibility}")
            c.drawString(50, y - 60, f"Credibility: {credibility}")
            c.drawString(50, y - 80, f"Competitiveness: {competitiveness}")
            c.drawString(50, y - 100, f"Operating Risk: {operating_risk}")

            # Prediction Result
            c.setFont("Helvetica-Bold", 14)
            if prediction == 1:
                c.setFillColorRGB(1, 0, 0)
                c.drawString(50, y - 140, f"⚠️ Bankruptcy Risk: HIGH ({probability:.2f}%)")
            else:
                c.setFillColorRGB(0, 0.6, 0)
                c.drawString(50, y - 140, f"✅ Bankruptcy Risk: LOW ({probability:.2f}%)")

            # Add Pie Chart Image to PDF ✅
            pie_img = ImageReader(pie_buffer)
            c.drawImage(pie_img, 150, 150, width=300, height=300)

            # Footer
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Oblique", 9)
            c.drawString(50, 40, "Generated by Bankruptcy Prediction App |Amit Choudhury PC")

            c.showPage()
            c.save()

            pdf_value = pdf_buffer.getvalue()
            pdf_buffer.close()
            pie_buffer.close()

            # Download button
            st.download_button(
                label="📄 Download PDF Report (With Pie Chart)",
                data=pdf_value,
                file_name="Bankruptcy_Report_With_Pie.pdf",
                mime="application/pdf"
            )
#Steps-7
#================================= MODEL BUILDING PAGE ====================================
elif menu == "Model Building":
    st.title("🧠 Model Building & Evaluation")

    uploaded_file = st.file_uploader("📤 Upload your dataset file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Handle both Excel and CSV
        file_name = uploaded_file.name
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported file format. Please upload CSV or XLSX.")
            st.stop()

        st.success("✅ File uploaded successfully!")
        st.write("### 🔍 Dataset Preview")
        st.dataframe(df.head())

        target_col = st.selectbox("🎯 Select Target Column", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode target if it’s categorical
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_name = st.selectbox(
            "🤖 Choose a Model",
            [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "K-Nearest Neighbors (KNN)",
                "Support Vector Machine (SVM)",
                "Gradient Boosting",
                "XGBoost",
            ],
        )

        if st.button("🚀 Train Model"):
            st.info(f"Training {model_name}... Please wait.")

            # Import models
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC

            # Try to import XGBoost
            try:
                from xgboost import XGBClassifier
                xgb_available = True
            except ImportError:
                xgb_available = False

            # Choose model
            if model_name == "Logistic Regression":
                clf = LogisticRegression(max_iter=1000)
            elif model_name == "Decision Tree":
                clf = DecisionTreeClassifier(random_state=42)
            elif model_name == "Random Forest":
                clf = RandomForestClassifier(random_state=42)
            elif model_name == "K-Nearest Neighbors (KNN)":
                clf = KNeighborsClassifier()
            elif model_name == "Support Vector Machine (SVM)":
                clf = SVC(probability=True)
            elif model_name == "Gradient Boosting":
                clf = GradientBoostingClassifier(random_state=42)
            elif model_name == "XGBoost":
                if xgb_available:
                    clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
                else:
                    st.warning("⚠️ XGBoost not installed. Install it with `pip install xgboost`.")
                    st.stop()

            # Train model
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Results
            st.subheader("📈 Model Performance")
            st.write(f"**Accuracy:** {acc*100:.2f}%")

            st.text("📋 Classification Report:")
            st.text(classification_report(y_test, y_pred))

            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            # Save trained model
            with open("Bankruptcy_Model_Final.pkl", "wb") as f:
                pickle.dump(clf, f)

            st.success(f"✅ {model_name} trained and saved successfully!")

#Steps-8
#================================ PROJECT INFORMATION PAGE ====================================
elif menu == "Project Info":
    st.title("📘 Project Details")
    st.write("""
    **Project Title:** Bankruptcy Prediction using Machine Learning  
    **Course:** Data Science  
    **Organization:** EXCELR  
    **Dataset:** Financial Risk Indicators  
    **Goal:** Build a prediction model that identifies companies at risk of bankruptcy.
    ---
    **Tech Stack Used:**  
    - Python, Pandas, NumPy  
    - Machine Learning: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM),Gradient      Boosting,XGBoost.  
    - Deployment: Streamlit
    """)
#Steps-9
# ============================== ABOUT DEVELOPER PAGE =========================================
elif menu == "About Developer":
    st.title("👨‍💻 Developer Info")
    st.write("""
    **Team Name:** Amit Choudhury  
    **Role:** Data Scientist  
    **Skills:** Python, ML, SQL, Streamlit, Tableau, Excel, ChatGPT  
    **Vision:** Creating AI solutions to solve real business challenges 💡
    """)
    st.markdown("---")
    st.write("Thank you for visiting my project! 🚀")


# In[ ]:




