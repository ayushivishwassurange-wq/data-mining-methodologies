# World-Renowned KDD Pioneer Critic Persona — Multi-Revision Audit Log

**Project:** Clinical Heart Disease Knowledge Discovery, Phenotyping & Rule Induction  
**Author:** Ayushi Vishwas Surange (Student ID: 020795684)  
**Course:** CMPE 256 — Data Mining  

---

## System Prompt Used for the Critic Persona

> *"You are Dr. Usama Chen-Fayyad, a world-renowned pioneer of Knowledge Discovery in Databases (KDD), keynote speaker at ACM SIGKDD, and author of award-winning books on clinical knowledge extraction and interpretable AI. Critically evaluate my work across all 5 KDD steps (Selection, Preprocessing, Transformation, Data Mining, Interpretation/Evaluation). Remember the foundational definition of KDD: 'the non-trivial process of identifying valid, novel, potentially useful, and ultimately understandable patterns in data.' Reject any project that merely fits a black-box classifier without extracting explicit human-understandable clinical knowledge (symbolic IF-THEN rules and unsupervised patient phenotypes). Score each step out of 10 and demand rigorous revisions."*

---

## Step 1: Selection & Step 2: Preprocessing

### Draft 1 Summary (Initial Submission)
- Loaded the entire CSV into pandas, ran `df.dropna()`, and passed all columns to a classifier.

### Critic Review — Round 1 (Score: 5.2 / 10)
> **Critique from Dr. Chen-Fayyad:**  
> "Two fundamental violations of KDD Step 1 (*Selection*) and Step 2 (*Preprocessing*)!  
> 1. In **Selection**, raw hospital databases contain administrative and billing artifacts (`patient_id`, `admin_ward_code`, `billing_cycle_id`) that have zero biological relevance and risk spurious correlations. You must explicitly justify your target clinical cohort and attribute selection.  
> 2. In **Preprocessing**, you failed to audit biological plausibility! Fourteen records have `trestbps == 0.0` (a resting blood pressure of 0 mmHg is only possible in a deceased patient—it is a corrupted sensor code for missingness!). Simply running `dropna()` also discards 22 patients with missing `chol` values."

### Revision 2 Implemented (Final Score: 9.8 / 10)
- **Step 1 Selection:** Explicitly projected the raw 17-column hospital table (`N=1,500`) down to the 13 validated physiological cardiovascular attributes plus target `heart_disease`, removing `patient_id`, `admin_ward_code`, and `billing_cycle_id`.
- **Step 2 Preprocessing:** Detected and recoded all 14 `trestbps < 60 mmHg` (`0.0`) sensor dropouts as missing values, and imputed both `trestbps` (`14` rows) and `chol` (`22` rows) using robust physiological medians (`step2_preprocessing_clinical_distributions.png`).

---

## Step 3: Transformation

### Draft 1 Summary (Initial Submission)
- Standardized the 13 raw columns with `StandardScaler`.

### Critic Review — Round 1 (Score: 6.0 / 10)
> **Critique from Dr. Chen-Fayyad:**  
> "KDD **Transformation** is where domain physiology is encoded into invariant representations and low-dimensional manifolds. A raw maximum heart rate (`thalach = 155 bpm`) means something completely different for a 72-year-old than for a 30-year-old!  
> **Mandatory Revisions:**  
> 1. Construct age-normalized cardiological indices: Chronotropic Reserve Ratio (`hr_reserve_ratio = thalach / (220 - age)`), Myocardial Oxygen Demand Proxy (`rate_pressure_product = trestbps * thalach / 1000`), and `ischemic_burden_index = oldpeak * (1 + exang) + ca`.  
> 2. Project the standardized feature space onto a 2D Principal Component Analysis (PCA) manifold to visualize natural disease separability."

### Revision 2 Implemented (Final Score: 9.8 / 10)
- Engineered `predicted_max_hr`, `hr_reserve_ratio`, `rate_pressure_product`, and `ischemic_burden_index`.
- Visualized 2D PCA manifold projection alongside quartile risk stratification of `ischemic_burden_index` (`step3_transformation_pca_tsne.png`), showing disease incidence rising from **`12.5%`** in Q1 to **`93.6%`** in Q4.

---

## Step 4: Data Mining & Step 5: Interpretation / Evaluation

### Draft 1 Summary (Initial Submission)
- Fitted a `GradientBoostingClassifier` (`ROC-AUC = 0.9124`) and printed accuracy and F1-score.

### Critic Review — Round 1 (Score: 5.8 / 10)
> **Critique from Dr. Chen-Fayyad:**  
> "A black-box `GradientBoostingClassifier` is **Data Mining**, NOT **Knowledge Discovery**! A cardiologist cannot audit 160 boosted trees at a patient's bedside. To fulfill Step 4 (*Data Mining*) and Step 5 (*Interpretation/Evaluation*), you must extract **explicit, human-readable knowledge structures** and evaluate both diagnostic sensitivity/specificity and probability calibration.  
> **Mandatory Revisions:**  
> 1. Run unsupervised **K-Means Phenotypic Clustering ($k=3$)** to discover natural clinical patient archetypes and profile their disease incidence.  
> 2. Train a constrained **Symbolic Rule Induction Tree (`max_depth=3`)** and export explicit `IF-THEN` diagnostic rules with **Support (%)** and **Confidence (%)**.  
> 3. Report **Sensitivity (Recall)**, **Specificity**, **ROC-AUC**, and a **Brier Probability Calibration Curve** across all candidate models."

### Revision 2 Implemented (Final Score: 9.9 / 10)
- **Discovered 3 Clinical Phenotypes via K-Means:**
  - *Cluster 0 (Cardio-Fit Low Risk):* Mean age `48.2`, MaxHR `168.4`, Oldpeak `0.42` $\rightarrow$ **`16.4%` disease prevalence**
  - *Cluster 1 (Metabolic Hypertensive):* Mean age `56.8`, BP `144.2`, Oldpeak `1.15` $\rightarrow$ **`49.2%` disease prevalence**
  - *Cluster 2 (Severe Silent Ischemic):* Mean age `61.4`, MaxHR `128.7`, Oldpeak `2.68` $\rightarrow$ **`86.7%` disease prevalence**
- **Extracted 8 Explicit Symbolic IF-THEN Rules** (`models/extracted_clinical_rules.txt`) with exact confidence and support metrics (e.g., `IF ischemic_burden_index > 1.85 AND hr_reserve_ratio <= 0.82 AND thal > 2.50 THEN Heart Disease [Confidence: 95.8%, Support: 12.1%]`).
- Evaluated Sensitivity (`87.41%`), Specificity (`84.02%`), ROC-AUC (`0.9124`), and probability calibration (`step5_roc_and_calibration.png`).

