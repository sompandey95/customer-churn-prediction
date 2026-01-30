📉 Customer Churn Prediction & Risk Analysis System
An end-to-end Machine Learning project to predict customer churn risk and provide actionable business insights through an interactive Streamlit dashboard.

📌 Problem Statement
• Customer churn directly impacts revenue and growth. Retaining existing customers is significantly cheaper than acquiring new ones.
• This project aims to identify customers at risk of churning and classify them into risk levels so that businesses can take proactive retention actions.

🎯 Project Objectives
• Analyze customer behavior and churn patterns
• Build a robust churn prediction model
• Handle class imbalance effectively
• Compare multiple ML models
• Deploy the final model using an interactive Streamlit dashboard
• Present predictions as business-friendly risk scores

🧠 Solution Overview
Workflow:
Data Collection → EDA → Feature Engineering → Model Training
→ Model Evaluation → Risk Calibration → Streamlit Deployment

📂 Project Structure
customer-churn-system/
│
├── data/
│   ├── raw_churn_data.csv
│   └── processed_churn_data.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── models/
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md

📊 Dataset
• Source: Telco Customer Churn Dataset
• Records: 7,043 customers
• Target Variable: Churn (Yes / No)
• Features: Contract type, tenure, payment method, charges, services, etc.

🔍 Exploratory Data Analysis (EDA)
• Key insights derived from EDA:
• Customers on month-to-month contracts churn significantly more
• Low-tenure customers are at the highest risk
• Higher monthly charges correlate with increased churn
• Churn data is imbalanced, requiring careful metric selection

⚙️ Feature Engineering
• Dropped non-predictive identifiers (customerID)
• Converted TotalCharges to numeric and handled missing values
• One-hot encoded categorical variables
• Scaled numerical features using StandardScaler
• Saved preprocessing artifacts for deployment consistency

🤖 Model Training & Evaluation
Models Used:
• Logistic Regression (Baseline & Final Model)
• Random Forest Classifier (Comparison Model)

Evaluation Metrics:
• Precision
• Recall
• F1-Score
• Confusion Matrix

Model Selection Rationale:
• Logistic Regression achieved higher recall for churned customers
• Recall was prioritized to minimize false negatives
• Outputs were treated as risk scores, not absolute probabilities

📌 Risk Calibration Strategy
Instead of binary churn labels, predictions are interpreted as risk bands:

Risk Score Range	Risk Level
< 0.45	Low Risk
0.45 – 0.60	Medium Risk
> 0.60	High Risk

This aligns model output with real-world business decision-making.

🖥️ Streamlit Dashboard Features
• User-friendly input form for customer details
• Real-time churn risk prediction
• Business-oriented risk classification (Low / Medium / High)
• Actionable retention recommendations
• End-to-end ML deployment pipeline

🚀 How to Run the Project
1️⃣ Clone Repository
git clone <your-github-repo-link>
cd customer-churn-system

2️⃣ Create & Activate Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit App
streamlit run app/streamlit_app.py

🧠 Key Learnings
• Importance of aligning training and inference pipelines
• Handling imbalanced datasets in classification problems
• Treating ML outputs as decision support signals
• Deploying ML models with business-oriented interpretation

📈 Future Improvements
• Add more customer attributes in the Streamlit UI
• Implement probability calibration techniques
• Add SHAP-based explainability
• Store predictions and user sessions

👤 Author
Som Pandey
B.Tech CSE (2026)
Aspiring Data Analyst / ML Engineer