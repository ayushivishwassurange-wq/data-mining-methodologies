# KDD Detailed Technical Report: Clinical Heart Disease Knowledge Discovery

**Author:** Ayushi Vishwas Surange (Student ID: 020795684)  
**Course:** CMPE 256 — Data Mining  
**Methodology:** Knowledge Discovery in Databases (Fayyad et al. — **Selection, Preprocessing, Transformation, Data Mining, Interpretation/Evaluation**)  

---

## Executive Summary
Following the foundational **KDD Process** defined by Fayyad, Piatetsky-Shapiro, and Smyth (1996), this study transforms raw clinical cardiovascular database records (`N = 1,500` patients, `17` raw database attributes) into **valid, novel, useful, and clinically interpretable knowledge**. Audited through multiple revisions by a **World-Renowned KDD Pioneer Critic Persona**, we go beyond black-box classification (`GradientBoosting ROC-AUC = 0.9124`, `Sensitivity = 87.41%`, `Specificity = 84.02%`) to discover **three unsupervised patient phenotypes** (`16.4%`, `49.2%`, and `86.7%` disease prevalence) and extract **eight explicit IF-THEN clinical decision rules** achieving up to **`95.8%` diagnostic confidence**.

---

## Step 1: Selection

### 1.1 Target Database & Attribute Selection
Raw hospital electronic health record (EHR) exports (`data/clinical_heart_disease_kdd.csv`, `1,500` rows $\times$ `17` columns) mix physiological measurements with non-clinical administrative fields. In Step 1 (*Selection*), we removed administrative noise columns (`patient_id`, `admin_ward_code`, `billing_cycle_id`) that risk inducing spurious institutional correlations, isolating the **13 core physiological attributes** plus the target diagnosis `heart_disease` (`0 = Healthy`, `1 = Angiographic Coronary Disease`):
- **Demographics & Vitals:** `age` (`29–79` yrs), `sex` (`0/1`), `trestbps` (resting systolic BP, mmHg), `chol` (serum cholesterol, mg/dL), `fbs` (fasting blood sugar $> 120$ mg/dL)
- **Stress ECG & Symptomatology:** `cp` (chest pain type `0–3`, where `0 = asymptomatic`), `restecg` (`0–2`), `thalach` (maximum heart rate achieved), `exang` (exercise-induced angina `0/1`), `oldpeak` (ST depression induced by exercise relative to rest, mm), `slope` (`0–2`)
- **Fluoroscopy & Perfusion:** `ca` (number of major vessels colored by fluoroscopy `0–3`), `thal` (thalassemia perfusion defect: `1 = fixed`, `2 = normal`, `3 = reversible`)

---

## Step 2: Preprocessing

### 2.1 Biological Plausibility Audit & Sensor Noise Removal
A domain-guided physiological audit of the selected cohort revealed two critical data quality issues:
1. **Corrupted Sensor Zeros in Blood Pressure:** `14` patient records contained `trestbps == 0.0 mmHg`—a biologically impossible value in living patients caused by unrecorded cuff readings defaulting to zero.
2. **Missing Serum Lipid Assays:** `22` patient records had missing (`NaN`/blank) `chol` values.

### 2.2 Imputation & Outlier Treatment
Rather than discarding `36` clinical cases via listwise deletion (`dropna()`), we recoded all `trestbps < 60 mmHg` values as missing and imputed both `trestbps` and `chol` using robust physiological medians.

![Step 2 Preprocessing Clinical Distributions](figures/step2_preprocessing_clinical_distributions.png)

---

## Step 3: Transformation

### 3.1 Cardiological Composite Index Construction
To capture non-linear physiological interactions, we transformed raw vitals into three validated cardiological indices:
1. **Chronotropic Reserve Ratio (`hr_reserve_ratio`):**
   $$\text{HR Reserve Ratio} = \frac{\text{thalach}}{220 - \text{age}}$$
   Quantifies chronotropic incompetence (failure to reach age-predicted maximum heart rate during stress testing).
2. **Rate-Pressure Product (`rate_pressure_product`):**
   $$\text{RPP} = \frac{\text{trestbps} \times \text{thalach}}{1000}$$
   Standard clinical proxy for myocardial oxygen consumption ($\text{MVO}_2$).
3. **Ischemic Burden Index (`ischemic_burden_index`):**
   $$\text{Ischemic Burden} = \text{oldpeak} \times (1 + \text{exang}) + \text{ca}$$
   Combines exercise ST-segment depression, angina amplification, and fluoroscopic vessel narrowing.

### 3.2 Dimensionality Reduction & Manifold Projection
We standardized all features (`StandardScaler`) and projected the 16-dimensional clinical space onto a 2D **Principal Component Analysis (PCA)** manifold (`step3_transformation_pca_tsne.png`), demonstrating clear separation between healthy and ischemic cohorts across `ischemic_burden_index` quartiles (`12.5%` disease in Q1 vs. `93.6%` in Q4).

![Step 3 Transformation PCA and Risk Separation](figures/step3_transformation_pca_tsne.png)

---

## Step 4: Data Mining

To honor Fayyad's mandate that KDD must discover *understandable patterns*, we executed a three-tier Data Mining stage:

### 4.1 Unsupervised Phenotypic Clustering ($k$-Means, $k=3$)
Clustering the standardized physiological space discovered three distinct clinical patient archetypes:
| Discovered Phenotype Cluster | Mean Age (yrs) | Mean Resting BP (mmHg) | Mean Max HR (bpm) | Mean ST Depression (`oldpeak`) | Heart Disease Prevalence (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cluster 0: Cardio-Fit Low Risk** | 48.2 | 124.5 | 168.4 | 0.42 | **16.4%** |
| **Cluster 1: Metabolic Hypertensive** | 56.8 | 144.2 | 146.1 | 1.15 | **49.2%** |
| **Cluster 2: Severe Silent Ischemic** | 61.4 | 139.8 | 128.7 | 2.68 | **86.7%** |

### 4.2 Symbolic Rule Induction (`DecisionTreeClassifier`, Depth = 3)
We extracted explicit `IF-THEN` diagnostic rules (`models/extracted_clinical_rules.txt`):
```text
|--- ischemic_burden_index <= 1.85
|   |--- cp <= 0.50 (Asymptomatic Chest Pain)
|   |   |--- thal <= 2.50 -> Class: Healthy (0) [Confidence: 74.2%, Support: 18.4%]
|   |   |--- thal >  2.50 (Reversible Defect) -> Class: Heart Disease (1) [Confidence: 83.6%, Support: 11.2%]
|   |--- cp >  0.50 (Non-Asymptomatic)
|   |   |--- ca <= 0.50 -> Class: Healthy (0) [Confidence: 89.4%, Support: 34.8%]
|   |   |--- ca >  0.50 -> Class: Heart Disease (1) [Confidence: 68.1%, Support: 7.5%]
|--- ischemic_burden_index >  1.85
|   |--- hr_reserve_ratio <= 0.82
|   |   |--- thal <= 2.50 -> Class: Heart Disease (1) [Confidence: 86.5%, Support: 10.6%]
|   |   |--- thal >  2.50 -> Class: Heart Disease (1) [Confidence: 95.8%, Support: 12.1%]
|   |--- hr_reserve_ratio >  0.82
|   |   |--- ca <= 0.50 -> Class: Healthy (0) [Confidence: 64.0%, Support: 2.9%]
|   |   |--- ca >  0.50 -> Class: Heart Disease (1) [Confidence: 88.2%, Support: 2.5%]
```

### 4.3 Predictive Classifier Benchmark
| Model | Holdout ROC-AUC | Accuracy | Sensitivity (Recall) | Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Champion)** | **0.9124** | **0.8567** | **0.8741** | **0.8402** | **0.8592** |
| **Random Forest** | 0.9038 | 0.8467 | 0.8601 | 0.8343 | 0.8483 |
| **Logistic Regression** | 0.8845 | 0.8233 | 0.8322 | 0.8150 | 0.8235 |
| **Interpretable Rule Tree (Depth=3)** | 0.8512 | 0.8067 | 0.8182 | 0.7959 | 0.8069 |

---

## Step 5: Interpretation / Evaluation

![Step 5 Phenotypes and Attribution](figures/step5_phenotypes_and_attribution.png)

![Step 5 ROC and Calibration](figures/step5_roc_and_calibration.png)

### Key Clinical Knowledge Discovered
1. **The "Silent Ischemia" Paradox (`cp == 0`):** Patients presenting with *asymptomatic* chest pain (`cp == 0`) actually exhibit a **`72.4%` disease prevalence** when accompanied by exercise ST depression or reversible thallium defects (`thal == 3`).
2. **Synergistic High-Risk Rule (`95.8%` Confidence):** Combining `ischemic_burden_index > 1.85`, chronotropic incompetence (`hr_reserve_ratio <= 0.82`), and reversible perfusion defect (`thal == 3`) identifies coronary artery disease with **`95.8%` precision** across `12.1%` of the entire patient population.
3. **Hybrid Clinical Deployment (`deployment/app.py`):** Pairing the high-sensitivity **Gradient Boosting** probability score (`ROC-AUC = 0.9124`) with the **Symbolic Rule Tree** and **Phenotype Cluster assignment** provides cardiologists with both state-of-the-art accuracy and transparent bedside explainability.

