# ✈️ Airbnb Booking Destination Prediction: End-to-End ML Pipeline

## 📌 Business Problem
Airbnb wants to predict the first booking destination of new users. Predicting this early allows the marketing team to personalize advertisements, increasing conversion rates and reducing wasted ad spend on irrelevant regions.

## 🛠️ The Data & Challenges
The raw data for this project is provided by Airbnb and can be downloaded from the [Kaggle Competition Page](https://www.kaggle.com/competitions/airbnb-recruiting-new-user-bookings/data). 

The dataset consisted of over 270,000 users and a massive **10 million+ row session log**.
* **High Cardinality & Sparsity:** Web action logs required aggressive aggregation and dimensionality reduction to prevent memory crashes.
* **Target Imbalance:** ~60% of users did not book (NDF), requiring synthetic data balancing and stratified evaluation.
* **Ghost Users:** Cleaned missing identifiers caused by Ad Blockers and Incognito browsing.

## 🧠 Modeling Approach
I engineered user-level features from the session logs and tested multiple tree-based algorithms. Standard accuracy was discarded in favor of **NDCG@5**, which is optimized for recommendation engines.
* **Baseline:** Logistic Regression & Random Forest (Rejected due to severe overfitting: 94% Train / 79% CV).
* **Final Model:** XGBoost Classifier.
* **Optimization:** Used **Optuna** for Bayesian hyperparameter tuning (optimizing the Learning Rate/Eta and n_estimators) combined with **Stratified K-Fold (k=3)** to handle the imbalanced target.
* **Final Kaggle Score:** ~0.8799 (NDCG@5).

## 📊 Key Business Insights
1. **Ad Spend Optimization:** Bookings drop drastically for countries >9,000 km away (e.g., Australia). Recommended cutting ad spend for distant outliers and reallocating budget to Core European Markets (FR, DE, GB).
2. **UX Prioritization:** ~74% of international trips go to non-English speaking destinations. Recommended the web team prioritize automated translation tools.

## 💻 Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** XGBoost, Scikit-Learn, Optuna, SMOTE
* **Visualization:** Matplotlib, Seaborn
