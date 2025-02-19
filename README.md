# SyriaTel Customer Churn Prediction

This project aims to predict customer churn for SyriaTel, a telecommunications company, using machine learning classification models. The dataset contains customer data and service usage information. The goal is to identify at-risk customers and provide insights for retention strategies.

## Tech Stack
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-learn (SMOTE)

## Exploratory Data Analysis (EDA)

For modeling, the SyriaTel customer dataset was cleaned, analyzed, and preprocessed. Missing values were imputed, and categorical data was one-hot encoded. Outliers were handled using the Interquartile Range (IQR) method.

The dataset showed an **imbalance in churn distribution**, with far more non-churners than churners. To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** was used to balance the training dataset.

### Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)

## Feature Engineering

The following features were selected for modeling:
- **Numerical Features**: Account length, total day minutes, total evening minutes, total night minutes, customer service calls, etc.
- **Categorical Features**: International plan, voice mail plan.
- **Engineered Features**: One-hot encoding for categorical variables and standardization for numerical variables.

## Modeling

Three different machine learning models were implemented and evaluated:

### 1. Logistic Regression
**Best Parameters:** `C = 1`
- **Accuracy:** 73.3%
- **Precision & Recall:**
  - Non-churners (Class 0): **Precision = 94%, Recall = 74%**
  - Churners (Class 1): **Precision = 32%, Recall = 71%**
- **Insights:**
  - Logistic Regression struggles with imbalanced data, leading to **low precision for churners (32%)**.
  - However, recall for churners is **high (71%)**, meaning the model correctly identifies most churners.
  - Not ideal for precise churn prediction due to high false positive rates.

### 2. Decision Tree
**Best Parameters:** `max_depth = 6, min_samples_split = 2`
- **Accuracy:** 88.9%
- **Precision & Recall:**
  - Non-churners: **Precision = 95%, Recall = 92%**
  - Churners: **Precision = 60%, Recall = 73%**
- **Insights:**
  - Decision trees capture complex relationships better than logistic regression.
  - Recall remains high for churners (**73%**), but precision improves to **60%**.
  - The model is prone to overfitting without proper pruning.

### 3. Random Forest (Best Performing Model)
**Best Parameters:** `max_depth = 20, min_samples_split = 2, n_estimators = 50`
- **Accuracy:** 91.5%
- **Precision & Recall:**
  - Non-churners: **Precision = 95%, Recall = 95%**
  - Churners: **Precision = 70%, Recall = 72%**
- **Insights:**
  - **Random Forest performed the best**, with the highest accuracy and a good balance between precision and recall.
  - It reduces overfitting by averaging multiple decision trees.
  - Precision for churners improves to **70%**, reducing false positives compared to the decision tree.
  
## Visualization
- **Decision Tree Visualization**: The decision tree model was visualized to illustrate key decision splits.
- **Feature Importance Graph**: Random Forest feature importance was plotted to highlight the most significant variables.
- **Classification Report & Confusion Matrix**:
  ![Random Forest Classification Report](images/random_forest_classification_report.png)
  ![Random Forest Confusion Matrix](images/random_forest_confusion_matrix.png)

## Feature Importance
The most important features influencing churn prediction were:
1. **Total Day Minutes** (higher usage correlates with churn)
2. **Customer Service Calls** (frequent complaints indicate dissatisfaction)
3. **International Plan** (customers with this plan tend to churn more)

## Conclusions / Next Steps
- **Hyperparameter tuning**: Further tuning of tree-based models (e.g., increasing n_estimators) could improve performance.
- **Additional models**: Testing boosting models like **XGBoost** or **LightGBM** for better churn prediction.
- **Customer segmentation**: Cluster customers into different risk groups based on their churn probability.
- **Business recommendations**: Develop proactive customer retention strategies based on high-risk features (e.g., offering better customer service, targeted promotions for high-usage customers).

---
This project provides valuable insights for SyriaTel to reduce churn and improve customer satisfaction using machine learning techniques.
