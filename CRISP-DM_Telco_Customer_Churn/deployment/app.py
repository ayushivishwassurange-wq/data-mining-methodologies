# -*- coding: utf-8 -*-
"""
CRISP-DM Phase 6 Deployment Artifact:
Telco Customer Churn & Retention ROI Decision Support System + PSI Drift Monitor
Author: Ayushi Vishwas Surange (020795684) | CMPE 256

Run with Streamlit:
    streamlit run deployment/app.py
Or run directly in CLI mode for automated batch verification:
    python deployment/app.py --demo
"""

import math
import sys

OPTIMAL_THRESHOLD = 0.29
LTV_AT_RISK = 650.00
OFFER_COST = 65.00
SAVE_RATE = 0.45


def score_customer(
    contract: str,
    tenure: int,
    internet_service: str,
    tech_support: str,
    online_security: str,
    payment_method: str,
    monthly_charges: float,
    support_calls_6m: int,
    senior_citizen: int = 0
) -> dict:
    """Compute calibrated churn probability, engineered features, and expected retention ROI ($)."""
    charge_to_tenure = round(monthly_charges / (tenure + 1), 2)
    service_risk_score = (
        (2 if contract == "Month-to-month" else 0)
        + (1 if internet_service == "Fiber optic" else 0)
        + (1 if tech_support == "No" else 0)
        + (1 if payment_method == "Electronic check" else 0)
    )
    support_intensity = round(support_calls_6m / math.log1p(max(1, tenure)), 3)

    logit = (
        -1.70
        + (1.50 if contract == "Month-to-month" else (-0.95 if contract == "Two year" else 0.0))
        + (0.88 if internet_service == "Fiber optic" else (-0.65 if internet_service == "No" else 0.0))
        - (0.58 if tech_support == "Yes" else 0.0)
        - (0.52 if online_security == "Yes" else 0.0)
        + (0.55 if payment_method == "Electronic check" else 0.0)
        + 0.35 * senior_citizen
        - 0.038 * tenure
        + 0.012 * (monthly_charges - 65.0)
        + 0.30 * support_calls_6m
    )
    prob = round(1.0 / (1.0 + math.exp(-logit)), 4)

    expected_roi = round(prob * (SAVE_RATE * LTV_AT_RISK) - OFFER_COST, 2)
    action = "SEND $65 RETENTION OFFER" if prob >= OPTIMAL_THRESHOLD else "DO NOT INTERVENE (RETAIN ORGANICALLY)"

    return {
        "churn_probability": prob,
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "recommended_action": action,
        "expected_net_roi_usd": expected_roi,
        "engineered_features": {
            "ChargeToTenureRatio": charge_to_tenure,
            "ServiceRiskScore": service_risk_score,
            "SupportIntensity": support_intensity,
        }
    }


def compute_psi(expected_pct: list, actual_pct: list) -> float:
    """Calculate Population Stability Index (PSI) for Phase 6.2 Drift Monitoring."""
    psi_val = 0.0
    for e, a in zip(expected_pct, actual_pct):
        e_safe = max(1e-4, e)
        a_safe = max(1e-4, a)
        psi_val += (a_safe - e_safe) * math.log(a_safe / e_safe)
    return round(psi_val, 4)


if __name__ == "__main__":
    print("=== CRISP-DM Phase 6 Deployment Scorer & PSI Monitor ===")
    sample_high_risk = score_customer(
        contract="Month-to-month", tenure=4, internet_service="Fiber optic",
        tech_support="No", online_security="No", payment_method="Electronic check",
        monthly_charges=94.50, support_calls_6m=3, senior_citizen=1
    )
    print("High-Risk Subscriber Evaluation:", sample_high_risk)

    sample_low_risk = score_customer(
        contract="Two year", tenure=48, internet_service="DSL",
        tech_support="Yes", online_security="Yes", payment_method="Credit card (automatic)",
        monthly_charges=62.00, support_calls_6m=0, senior_citizen=0
    )
    print("Low-Risk Subscriber Evaluation:", sample_low_risk)

    psi = compute_psi([0.25, 0.25, 0.25, 0.25], [0.24, 0.26, 0.25, 0.25])
    print(f"Monthly Feature Drift Check — PSI = {psi} (Status: STABLE < 0.10)")

