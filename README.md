
# 💳 Credit Risk Prediction using Machine Learning

## 🧠 Project Overview
This project aims to build a **data-driven predictive model** for credit card default risk analysis.  
By analyzing **customer demographics, repayment behavior, and transaction history**, the model predicts whether a client will **default next month**.  

The goal is to help **financial institutions identify high-risk customers**, improve decision-making, and **reduce non-performing assets (NPAs)**.

---

## 🎯 Objectives
1. Predict the likelihood of a customer defaulting on their next credit card payment.  
2. Identify key behavioral and financial factors influencing credit risk.  
3. Enable data-driven lending and credit management decisions.  

---

## 📂 Dataset Description
The dataset contains **30,000 records** with **25 attributes**, representing real-world credit card clients in Taiwan.  
Each record corresponds to one customer.

| Category | Description |
|-----------|-------------|
| **Demographic Attributes** | `Gender_Code`, `Education_Level`, `Marital_Status`, `Age_Years` |
| **Credit Attributes** | `Credit_Limit` (credit amount assigned to customer) |
| **Repayment Status** | `Repay_Sep` → `Repay_Apr` (payment delay history over 6 months) |
| **Billing Amounts** | `BillAmt_Sep` → `BillAmt_Apr` (monthly bill amounts) |
| **Payment Amounts** | `PaidAmt_Sep` → `PaidAmt_Apr` (actual monthly repayment amounts) |
| **Target Variable** | `Will_Default_Next_Month` (1 = Default, 0 = No Default) |

✅ No missing values were found.  
All columns are **numeric**, simplifying data preprocessing and modeling.

---

## 📊 Exploratory Data Analysis (EDA)
### 🔹 Correlation Insights
- **Strong positive correlation** among consecutive repayment and billing features.  
- `Repay_Sep` shows **0.32 correlation** with `Will_Default_Next_Month`, showing late payments directly increase default risk.  
- **Higher credit limits** are negatively correlated (-0.15) with defaults — financially stronger customers tend to repay regularly.  

### 🔹 Age & Credit Limit
- Default rate stable (20–30%) between **ages 25–55**.  
- Customers above 60 or below 25 show slightly higher default probability.  
- **Default probability decreases** as **credit limit increases**, showing that responsible clients get higher credit limits.

---

## ⚙️ Methodology

### **1. Data Preprocessing**
- Checked and confirmed **no missing values**.  
- Performed **feature scaling** using `StandardScaler`.  
- Applied **train-test split (80/20)** to validate model performance.  

### **2. Feature Engineering**
- Retained demographic and behavioral variables.  
- Dropped redundant correlated features (if any).  
- Focused on repayment and payment history as key predictors.  

### **3. Model Development**
Implemented seven supervised learning algorithms:
- Logistic Regression  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  

Each model was evaluated using cross-validation to ensure generalization.

### **4. Evaluation Metrics**
Models were compared using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  

---

## 🧮 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:---------:|:----------:|:-------:|:---------:|
| Logistic Regression | 0.716 | 0.726 | 0.716 | 0.721 |
| SVM | **0.816** | 0.798 | **0.817** | 0.792 |
| KNN | 0.793 | 0.770 | 0.793 | 0.775 |
| Decision Tree | 0.716 | 0.726 | 0.716 | 0.721 |
| Random Forest | 0.815 | 0.796 | 0.815 | 0.795 |
| **Gradient Boosting** | **0.818** | **0.801** | **0.818** | **0.797** |
| XGBoost | 0.810 | 0.790 | 0.810 | 0.791 |

🏆 **Best Model:** Gradient Boosting Classifier  
- Achieved **81.83% accuracy** with balanced precision and recall.  
- Outperformed other models in handling non-linear patterns and generalizing across test data.  

---

## 💡 Insights
1. **Repayment history** is the most influential factor in predicting defaults.  
2. **High credit limits** are linked with disciplined repayment.  
3. **Education and marital status** slightly influence repayment behavior.  
4. **Age group (25–55)** shows the most financial stability.  
5. **Gradient Boosting and SVM** deliver strong, interpretable, and scalable results.

---

## 🏁 Conclusion
This project successfully demonstrates how machine learning can be applied for **credit risk analysis**.  
The model can:
- Detect potential defaulters early.  
- Help banks **optimize credit policies** and reduce risk exposure.  
- Support **data-driven financial decisions** and **responsible lending**.

---

## 🧰 Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost  
- **Algorithms:** Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost  
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC  

---

## 🚀 Future Enhancements
- Integrate **Deep Learning (ANNs)** for non-linear behavioral modeling.  
- Implement **SHAP or LIME** for model explainability.  
- Deploy using **Flask or Streamlit** for real-time risk predictions.  

---

## 📸 Visuals
*(Include these screenshots in your repo for clarity)*  
- Correlation Heatmap  
- Age vs Default Rate Plot  
- Credit Limit Distribution  
- Model Comparison Graph  

---

## 🧑‍💻 Author
**Developed by:** Ujwala Rapalli  
**GitHub Repository:** [credit-risk-predictor](https://github.com/ujwala-rapalli/credit-risk-predictor)
