# World-Renowned CRISP-DM Critic Persona — Multi-Revision Audit Log

**Project:** Telco Customer Churn Prediction & Proactive Retention ROI  
**Author:** Ayushi Vishwas Surange (Student ID: 020795684)  
**Course:** CMPE 256 — Data Mining  

---

## System Prompt Used for the Critic Persona

> *"You are Dr. Viktor Lindqvist, a world-renowned keynote speaker, enterprise data mining architect, and authority on the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology. You served on the original European CRISP-DM SIG consortium and have authored three award-winning textbooks on industrial data science lifecycle governance. Your job is to ruthlessly critique my work across all 6 phases and 24 subtasks of CRISP-DM. Do not accept textbook platitudes or shallow Kaggle-style notebooks. Demand explicit financial cost-benefit matrices, leakage-free data preparation pipelines, probability calibration, and concrete deployment drift monitoring. Score each draft out of 10 and list mandatory revisions before approving progression to the next phase."*

---

## Phase 1: Business Understanding

### Draft 1 Summary (Initial Submission)
- Stated goal: "Predict whether a telecom customer will churn (`Yes`/`No`) using machine learning models and maximize F1-score and Accuracy."

### Critic Review — Round 1 (Score: 5.5 / 10)
> **Critique from Dr. Lindqvist:**  
> "Unacceptable for an enterprise CRISP-DM Phase 1 deliverable. You have conflated a **Data Mining Goal** (Subtask 1.3: F1-score) with a **Business Objective** (Subtask 1.1). Telecom executives do not care about F1-scores; they care about **Net Retention Program ROI ($ USD)** and subscriber lifetime value (LTV). Furthermore, you completely omitted Subtask 1.2 (*Assess Situation*: assumptions, constraints, risks, and cost-benefit trade-offs).  
> **Mandatory Revisions:**  
> 1. Quantify the asymmetric dollar cost of a False Negative (lost LTV) vs. a False Positive (wasted promotional retention credit).  
> 2. Define the expected retention offer acceptance rate (`45%`) so Phase 5 can optimize a financial profit curve rather than a naive `0.50` probability cutoff."

### Revision 2 Implemented (Final Score: 9.8 / 10)
- Added explicit **Financial Cost-Benefit Matrix** in Subtask 1.1 & 1.2:
  - Customer Lifetime Value (LTV) at risk = **`$650.00`**
  - Proactive Retention Incentive Cost = **`$65.00`**
  - Retention Save Probability (given true churner receives offer) = **`45%`**
  - Net Benefit of True Positive (TP) = $0.45 \times \$650 - \$65 = +\$227.50$
  - Net Cost of False Positive (FP) = $-\$65.00$
- Added formal Risk & Contingency Register (concept drift from competitor pricing, cold-start new subscribers).

---

## Phase 2: Data Understanding

### Draft 1 Summary (Initial Submission)
- Plotted histograms of `tenure`, `MonthlyCharges`, and `TotalCharges`, plus a countplot of `Churn`.

### Critic Review — Round 1 (Score: 6.2 / 10)
> **Critique from Dr. Lindqvist:**  
> "Your EDA is descriptive but not diagnostic. In Subtask 2.4 (*Verify Data Quality*), you failed to explain *why* `TotalCharges` has 38 missing entries and failed to flag the 12 extreme anomalies in `MonthlyCharges` exceeding `$220/month` (when standard Fiber plans top out near `$118.75`).  
> **Mandatory Revisions:**  
> 1. Cross-tabulate missing `TotalCharges` against `tenure` and `Contract`.  
> 2. Perform formal outlier quantification (99th percentile bounds) on `MonthlyCharges`.  
> 3. Quantify churn interaction effects across `Contract` $\times$ `InternetService`."

### Revision 2 Implemented (Final Score: 9.7 / 10)
- Documented all 38 missing `TotalCharges` records and proved they can be deterministically reconstructed via `tenure * MonthlyCharges`.
- Flagged the 12 billing sensor/entry outliers (`$225–$295`) in `MonthlyCharges` for 99th-percentile Winsorization in Phase 3.
- Added bivariate conditional churn breakdown showing Month-to-Month accounts churn at **`>42%`** compared to **`<6%`** for Two-Year contracts.

---

## Phase 3: Data Preparation

### Draft 1 Summary (Initial Submission)
- Filled missing `TotalCharges` with `0`, applied `StandardScaler` to the entire dataset before `train_test_split`, and one-hot encoded categoricals.

### Critic Review — Round 1 (Score: 4.8 / 10)
> **Critique from Dr. Lindqvist:**  
> "Two critical methodological errors!  
> 1. **Data Leakage:** Fitting `StandardScaler` on the full dataset before splitting leaks test distribution statistics into training. Everything must be encapsulated inside a scikit-learn `ColumnTransformer` + `Pipeline`.  
> 2. **Naive Imputation & Zero Feature Construction (Subtask 3.3):** Imputing `TotalCharges = 0` for customers with positive tenure creates artificial leverage points. Moreover, raw `TotalCharges` is just a collinear product of `tenure` and `MonthlyCharges`. Where are your domain-constructed behavioral ratios?"

### Revision 2 Implemented (Final Score: 9.9 / 10)
- Eliminated data leakage by wrapping all imputation, scaling, and encoding inside a `ColumnTransformer` fitted strictly on `X_train`.
- Reconstructed missing `TotalCharges` using `tenure * MonthlyCharges` and clipped `MonthlyCharges` at the training 99th percentile.
- Constructed 4 high-signal domain features (Subtask 3.3):
  1. `ChargeToTenureRatio = MonthlyCharges / (tenure + 1)` (captures early bill shock)
  2. `AvgHistoricalMonthlySpend = TotalCharges / (tenure + 0.1)`
  3. `ServiceRiskScore` (additive structural risk index across Contract, Fiber, TechSupport, and Electronic Check)
  4. `SupportIntensity = SupportCallsLast6M / log1p(tenure)`

---

## Phase 4: Modeling

### Draft 1 Summary (Initial Submission)
- Trained a single `RandomForestClassifier` with default parameters and reported accuracy (`80.2%`).

### Critic Review — Round 1 (Score: 6.0 / 10)
> **Critique from Dr. Lindqvist:**  
> "With a `27.02%` base churn rate, a dummy model predicting 'No Churn' gets `72.98%` accuracy! Furthermore, because Phase 5 requires multiplying predicted probabilities by dollar values, your probabilities must be well-calibrated (`Brier Score`).  
> **Mandatory Revisions:**  
> 1. Benchmark at least 4 distinct algorithmic families (Linear, Bagging, Boosting, Neural Network) using 5-fold Stratified Cross-Validation.  
> 2. Report `CV_ROC_AUC`, `Test_ROC_AUC`, `Precision`, `Recall`, `F1_Score`, and **`Brier_Score`**."

### Revision 2 Implemented (Final Score: 9.8 / 10)
- Benchmarked `GradientBoostingClassifier` (`Test ROC-AUC = 0.8580`, `Brier = 0.1288`), `RandomForestClassifier` (`0.8517`), `MLPClassifier` (`0.8455`), and `LogisticRegression` (`0.8417`).
- Selected `GradientBoostingClassifier` as champion due to highest ROC-AUC and lowest Brier calibration error (`0.1288`).

---

## Phase 5 & Phase 6: Evaluation & Deployment

### Draft 1 Summary (Initial Submission)
- Plotted confusion matrix at default `0.50` threshold and saved `.pkl` file.

### Critic Review — Round 1 (Score: 6.5 / 10)
> **Critique from Dr. Lindqvist:**  
> "In Phase 5 (Subtask 5.1), you must prove that the model satisfies the **Phase 1 Business Objective**. What is the exact dollar savings at `p = 0.50` versus the profit-maximizing threshold $p^*$? And in Phase 6 (Subtask 6.2 *Plan Monitoring & Maintenance*), how will your production system detect covariate shift when telecom plan pricing changes next quarter?"

### Revision 2 Implemented (Final Score: 9.9 / 10)
- **Phase 5 Profit Curve Optimization:** Swept thresholds $p \in [0.05, 0.90]$. Found that shifting from `p = 0.50` (`$44,297.50` profit) to the optimal threshold **`p* = 0.29`** increases net retention profit on the 1,409-customer test cohort to **`$54,372.50`** (**`+$10,075.00` or `+22.7%` incremental profit gain**).
- **Phase 6 Production App & Drift Monitoring:** Built `deployment/app.py` (Streamlit interactive retention calculator + **Population Stability Index (PSI)** automated feature drift alert when $\text{PSI} > 0.20$) and `deployment/Dockerfile`.

