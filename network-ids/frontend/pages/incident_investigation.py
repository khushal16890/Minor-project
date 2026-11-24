"""
PAGE 4: Incident Investigation & Explainable AI (SHAP)
Provides deep-dive forensic analysis, feature attribution, and recommended automated response.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from frontend.api_client import APIClient


def render(api: APIClient):
    st.markdown("## 🔍 Incident Forensics & Explainable AI")
    st.markdown("Inspect granular incident evidence, SHAP feature attributions, and threat mitigation triggers.")

    # Fetch incident list for selection
    data = api.get_incidents(page=1, page_size=100)
    if not data or not data.get("incidents"):
        st.info("No security incidents recorded to investigate.")
        return

    incidents = data["incidents"]
    incident_options = {
        f"[{inc.get('severity', 'low').upper()}] {inc.get('attack_type', 'Unknown')} | {inc.get('src_ip', 'N/A')} -> {inc.get('dst_ip', 'N/A')} ({inc.get('id', '')[:8]})": inc.get("id")
        for inc in incidents
    }

    selected_label = st.selectbox("Select Incident to Investigate", list(incident_options.keys()))
    if not selected_label:
        return

    incident_id = incident_options[selected_label]
    incident = api.get_incident(incident_id)

    if not incident:
        st.error(f"Failed to fetch incident {incident_id}")
        return

    # Incident Overview Cards
    c1, c2, c3, c4 = st.columns(4)
    sev = incident.get("severity", "low")
    c1.metric("Threat Category", incident.get("attack_type", "Unknown"))
    c2.metric("Severity Level", sev.upper())
    c3.metric("Engine", incident.get("detection_method", "hybrid").upper())
    c4.metric("Risk Score", f"{incident.get('risk_score', 0):.1f} / 100")

    st.markdown("---")

    # Details & Traffic Meta
    col_meta, col_risk = st.columns([1, 1])

    with col_meta:
        st.subheader("Network Flow Telemetry")
        st.write(f"**Source IP / Port:** `{incident.get('src_ip', 'N/A')}:{incident.get('src_port', '')}`")
        st.write(f"**Destination IP / Port:** `{incident.get('dst_ip', 'N/A')}:{incident.get('dst_port', '')}`")
        st.write(f"**Protocol:** `{incident.get('protocol', 'N/A')}`")
        st.write(f"**ML Model Confidence:** `{incident.get('model_confidence', 0.0):.2%}`")
        st.write(f"**Isolation Forest Anomaly Score:** `{incident.get('anomaly_score', 0.0):.3f}`")
        if incident.get("signature_id"):
            st.write(f"**Suricata Signature ID:** `{incident.get('signature_id')}`")
            st.write(f"**Signature Message:** `{incident.get('signature_msg')}`")
        st.write(f"**Current Status:** `{incident.get('status', 'open')}`")

    with col_risk:
        st.subheader("Risk Assessment & Response")
        risk_score = incident.get("risk_score", 0.0)
        st.progress(min(1.0, max(0.0, risk_score / 100.0)))

        # Recommended action based on severity
        if sev == "critical":
            rec_action = "🚨 CRITICAL: Immediate containment recommended. Block source IP in firewall."
        elif sev == "high":
            rec_action = "⚠️ HIGH: Restrict network traffic. Admin approval recommended before blocking."
        elif sev == "medium":
            rec_action = "⚡ MEDIUM: Monitor host for recurring anomalous spikes. Generate ticket."
        else:
            rec_action = "ℹ️ LOW: Log event. Normal telemetry audit."

        st.info(rec_action)

        # Action Buttons
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🚫 Block Source IP", type="primary"):
                src_ip = incident.get("src_ip")
                if src_ip:
                    res = api.block_ip(
                        target_ip=src_ip,
                        reason=f"Incident {incident_id[:8]} ({incident.get('attack_type')})",
                        duration_minutes=60,
                        incident_id=incident_id
                    )
                    if res:
                        st.success(f"IP {src_ip} successfully blocked (simulated/enforced).")
                    else:
                        st.error(f"Failed to block IP {src_ip}.")
                else:
                    st.warning("No valid source IP found on this incident.")

        with b2:
            if st.button("✅ Mark as Resolved"):
                api.resolve_incident(incident_id, status="resolved", notes="Resolved from SOC Dashboard")
                st.success("Incident marked as resolved.")
                st.rerun()

    st.markdown("---")

    # SHAP Feature Explanations
    st.subheader("Explainable AI: Feature Attributions (SHAP)")
    shap_vals = incident.get("shap_values")
    if shap_vals and isinstance(shap_vals, dict):
        df_shap = pd.DataFrame(
            [{"Feature": k, "Contribution": float(v)} for k, v in shap_vals.items()]
        ).sort_values(by="Contribution", ascending=False)

        top_shap = df_shap.head(15)
        fig_shap = px.bar(
            top_shap,
            x="Contribution",
            y="Feature",
            orientation="h",
            color="Contribution",
            color_continuous_scale="RdBu_r",
            template="plotly_dark",
            title="Top Feature Contributions to Suspicious Classification"
        )
        fig_shap.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("SHAP values not available for this incident (e.g. signature detection or model not loaded during capture).")
