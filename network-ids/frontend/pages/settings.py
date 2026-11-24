"""
PAGE 7: System Settings & Engine Configuration
Customization of detection thresholds, risk scoring weights, and mitigation policies.
"""
import streamlit as st
from frontend.api_client import APIClient


def render(api: APIClient):
    st.markdown("## ⚙️ System Settings & Policies")
    st.markdown("Fine-tune scoring weights, firewall mitigation parameters, detection sensitivity, and operational modes.")

    settings = api.get_settings()
    if not settings:
        st.warning("Could not load system settings from backend.")
        return

    with st.form("settings_form"):
        # 1. Risk Scoring Weights
        st.subheader("1. Risk Scoring Engine Weights")
        st.markdown("Weights determine the contribution of each signal to the final 0–100 risk score.")
        rw = settings.get("risk_weights", {})
        c1, c2 = st.columns(2)
        with c1:
            w_conf = st.slider("Model Confidence Weight", 0.0, 1.0, float(rw.get("model_confidence", 0.40)), 0.05)
            w_sev = st.slider("Attack Severity Weight", 0.0, 1.0, float(rw.get("attack_severity", 0.30)), 0.05)
        with c2:
            w_anom = st.slider("Anomaly Score Weight", 0.0, 1.0, float(rw.get("anomaly_score", 0.20)), 0.05)
            w_freq = st.slider("Frequency Score Weight", 0.0, 1.0, float(rw.get("frequency_score", 0.10)), 0.05)

        total_weight = w_conf + w_sev + w_anom + w_freq
        if round(total_weight, 2) != 1.0:
            st.caption(f"⚠️ Total weight sum is {total_weight:.2f}. Recommended to sum to 1.00.")

        st.markdown("---")

        # 2. Mitigation Engine Policies
        st.subheader("2. Automated Mitigation & Firewall Policies")
        mit = settings.get("mitigation", {})
        m1, m2, m3 = st.columns(3)
        with m1:
            mit_enabled = st.checkbox("Enable Mitigation Engine", value=bool(mit.get("enabled", False)))
        with m2:
            mit_sim = st.checkbox("Simulation Mode (Safe)", value=bool(mit.get("simulation_mode", True)))
        with m3:
            mit_auto = st.checkbox("Auto-Block Critical Threats", value=bool(mit.get("auto_block_critical", False)))

        def_dur = st.number_input("Default Block Duration (Minutes)", 1, 1440, int(mit.get("default_block_duration_minutes", 60)))

        st.markdown("---")

        # 3. Detection Engine Thresholds
        st.subheader("3. Detection Engine Thresholds")
        det = settings.get("detection", {})
        d1, d2 = st.columns(2)
        with d1:
            ml_thresh = st.slider("ML Confidence Alert Threshold", 0.1, 1.0, float(det.get("ml_confidence_threshold", 0.70)), 0.05)
        with d2:
            anom_thresh = st.slider("Isolation Forest Anomaly Threshold", -1.0, 1.0, float(det.get("anomaly_threshold", -0.5)), 0.05)

        st.markdown("---")

        # Submit
        submitted = st.form_submit_button("💾 Save System Configuration", type="primary")
        if submitted:
            payload = {
                "risk_weights": {
                    "model_confidence": w_conf,
                    "attack_severity": w_sev,
                    "anomaly_score": w_anom,
                    "frequency_score": w_freq,
                },
                "mitigation": {
                    "enabled": mit_enabled,
                    "simulation_mode": mit_sim,
                    "auto_block_critical": mit_auto,
                    "default_block_duration_minutes": def_dur,
                },
                "detection": {
                    "ml_confidence_threshold": ml_thresh,
                    "anomaly_threshold": anom_thresh,
                }
            }
            res = api.update_settings(payload)
            if res:
                st.success("System configuration updated successfully!")
            else:
                st.error("Failed to update settings.")
