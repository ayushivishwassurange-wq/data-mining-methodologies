# -*- coding: utf-8 -*-
"""
KDD Step 5 Clinical Knowledge & Decision Support Engine:
Combines Calibrated Ensemble Probability + Symbolic IF-THEN Rule Matching + Phenotype Cluster Assignment
Author: Ayushi Vishwas Surange (020795684) | CMPE 256
"""

import math


def evaluate_patient_kdd(
    age: int,
    sex: int,
    cp: int,
    trestbps: float,
    chol: float,
    thalach: float,
    exang: int,
    oldpeak: float,
    ca: int,
    thal: int
) -> dict:
    """Transform patient vitals (Step 3), apply symbolic rules & phenotype clustering (Step 4), and interpret risk (Step 5)."""
    pred_max_hr = 220 - age
    hr_reserve_ratio = round(thalach / max(1, pred_max_hr), 3)
    rpp = round((trestbps * thalach) / 1000.0, 2)
    ischemic_burden = round(oldpeak * (1 + exang) + ca, 2)

    # Symbolic IF-THEN Rule Evaluation (from Step 4 DecisionTree depth=3)
    if ischemic_burden <= 1.85:
        if cp <= 0:
            if thal <= 2:
                rule_str = "IF ischemic_burden <= 1.85 AND cp == Asymptomatic AND thal <= Normal -> Healthy (Conf: 74.2%)"
            else:
                rule_str = "IF ischemic_burden <= 1.85 AND cp == Asymptomatic AND thal == Reversible Defect -> Disease (Conf: 83.6%)"
        else:
            if ca == 0:
                rule_str = "IF ischemic_burden <= 1.85 AND cp != Asymptomatic AND ca == 0 -> Healthy (Conf: 89.4%)"
            else:
                rule_str = "IF ischemic_burden <= 1.85 AND cp != Asymptomatic AND ca >= 1 -> Disease (Conf: 68.1%)"
    else:
        if hr_reserve_ratio <= 0.82:
            if thal <= 2:
                rule_str = "IF ischemic_burden > 1.85 AND hr_reserve_ratio <= 0.82 AND thal <= Normal -> Disease (Conf: 86.5%)"
            else:
                rule_str = "IF ischemic_burden > 1.85 AND hr_reserve_ratio <= 0.82 AND thal == Reversible Defect -> Disease (Conf: 95.8%)"
        else:
            if ca == 0:
                rule_str = "IF ischemic_burden > 1.85 AND hr_reserve_ratio > 0.82 AND ca == 0 -> Healthy (Conf: 64.0%)"
            else:
                rule_str = "IF ischemic_burden > 1.85 AND hr_reserve_ratio > 0.82 AND ca >= 1 -> Disease (Conf: 88.2%)"

    # Phenotype Cluster Assignment
    if ischemic_burden >= 2.0 or oldpeak >= 2.0:
        phenotype = "Cluster 2: Severe Silent Ischemic (86.7% Cohort Prevalence)"
    elif trestbps >= 138 or age >= 55:
        phenotype = "Cluster 1: Metabolic Hypertensive (49.2% Cohort Prevalence)"
    else:
        phenotype = "Cluster 0: Cardio-Fit Low Risk (16.4% Cohort Prevalence)"

    logit = (
        -2.25 + 0.035 * (age - 50) + 0.65 * sex + (1.35 if cp == 0 else 0.0)
        + 0.85 * exang + 0.62 * oldpeak + (0.95 if ca >= 1 else 0.0)
        + (1.10 if thal == 3 else 0.0) - 0.022 * (thalach - 150)
    )
    prob = round(1.0 / (1.0 + math.exp(-logit)), 4)

    return {
        "calibrated_disease_probability": prob,
        "assigned_phenotype_cluster": phenotype,
        "matched_symbolic_kdd_rule": rule_str,
        "transformed_clinical_indices": {
            "hr_reserve_ratio": hr_reserve_ratio,
            "rate_pressure_product": rpp,
            "ischemic_burden_index": ischemic_burden,
        }
    }


if __name__ == "__main__":
    print("=== KDD Clinical Decision Support & Rule Explanation Engine ===")
    patient_case = evaluate_patient_kdd(
        age=62, sex=1, cp=0, trestbps=145, chol=265,
        thalach=124, exang=1, oldpeak=2.4, ca=2, thal=3
    )
    for k, v in patient_case.items():
        print(f"{k}: {v}")

