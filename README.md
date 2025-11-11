🏦 Bankruptcy Prediction Project

📘 Overview
This End-to-End Machine Learning Project predicts the bankruptcy risk of a company based on key financial indicators.
It includes complete steps — from Exploratory Data Analysis (EDA) to Model Building, Evaluation, and Deployment using Streamlit.
Built using:

🐍 Python
📊 Pandas, NumPy, Matplotlib, Seaborn
🤖 Scikit-learn, XGBoost, Random Forest
🌐 Streamlit for interactive web app deployment
🎯 Business Objective
The main goal is to identify companies with a high likelihood of bankruptcy early, helping stakeholders make informed financial and operational decisions.

📂 Dataset Information
The dataset contains several financial and risk-based parameters such as:
Industrial Risk
Management Risk
Financial Flexibility
Credibility
Competitiveness
Operating Risk
The target variable represents whether a company is (Class)bankrupt (1) or non-bankrupt (0).

🔍 Exploratory Data Analysis (EDA)
Performed complete EDA to understand:
Data distribution and summary statistics
Correlation heatmap among features
Boxplots and histograms to visualize risk factors
Detection of outliers and data patterns

📈 Key Insight:
Companies with high industrial and management risk showed a significantly higher probability of bankruptcy.

🤖 Model Building and Evaluation
Multiple machine learning models were trained and compared:
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
XGBoost
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)

📊 Evaluation Metrics:
Accuracy
Confusion Matrix
ROC-AUC Curve
Cross-Validation Score
Best Performing Model: ✅ XGBoost (highest accuracy & AUC)

🌐 Streamlit Web App
An interactive web application allows users to:
Input company financial indicators
Get real-time bankruptcy prediction
View model probability and prediction results
Download results as PDF report

🧩 Run locally:
streamlit run Final_Dply.py

🧱 Project Structure
Bankruptcy_Prediction_Project/
│
├── Bankruptcy_Model_Final.pkl                # Trained model file
├── Bankruptcy (2).xlsx                       # Dataset
├── Bankruptcy_Prevention_EDA+MB_Amit.ipynb   # Jupyter Notebook (EDA + Model)
├── Final_Dply.py                             # Streamlit App Script
├── Bankruptcy_Presentation.pptx              # Final PPT Presentation
├── README.md                                 # Project Description

🚀 Future Enhancements
Add database integration for real company data
Deploy app on Streamlit Cloud 
Add automated retraining with new financial data
Improve UI with animations and dark theme

👨‍💻 Author
Amit Choudhury
Passionate in Data Science and Focused on Real-World ML Applications
📧 Email: [amitraychoudhury503@gmail.com]
🌐 Portfolio: [[GitHub Profile Link](https://github.com/amitraychoudhury/Bankruptcy_Prediction_Project)]
