"""
PAGE 3: Threat Detection
Comprehensive view of flagged security events, correlated hybrid threats, and alert triage.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from frontend.api_client import APIClient


def render(api: APIClient):
    st.markdown("## 🚨 Threat Detection & Correlated Alerts")
    st.markdown("Multi-engine detection stream combining Suricata signatures, supervised ML, and Isolation Forest anomaly analysis.")

    # Filter Bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sev_filter = st.selectbox("Severity", ["All", "critical", "high", "medium", "low"])
    with c2:
        method_filter = st.selectbox("Engine", ["All", "hybrid", "signature", "ml", "anomaly"])
    with c3:
        status_filter = st.selectbox("Status", ["All", "open", "investigating", "resolved"])
    with c4:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Alerts"):
            st.rerun()

    sev_arg = None if sev_filter == "All" else sev_filter
    method_arg = None if method_filter == "All" else method_filter
    status_arg = None if status_filter == "All" else status_filter

    incident_data = api.get_incidents(page=1, page_size=100, severity=sev_arg, detection_method=method_arg, status=status_arg)

    if not incident_data or not incident_data.get("incidents"):
        st.info("No security incidents match the current filters.")
        return

    incidents = incident_data["incidents"]
    df_inc = pd.DataFrame(incidents)

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Incidents Count", len(df_inc))
    crit_count = len(df_inc[df_inc["severity"] == "critical"]) if "severity" in df_inc.columns else 0
    m2.metric("Critical Alerts", crit_count)
    high_count = len(df_inc[df_inc["severity"] == "high"]) if "severity" in df_inc.columns else 0
    m3.metric("High Alerts", high_count)
    avg_conf = df_inc["model_confidence"].dropna().mean() if "model_confidence" in df_inc.columns else 0.0
    m4.metric("Avg ML Confidence", f"{avg_conf:.1%}")

    # Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Severity Distribution")
        sev_counts = df_inc["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        color_map = {"critical": "#ff4b4b", "high": "#ffa421", "medium": "#f0f2f6", "low": "#21c354"}
        fig_sev = px.bar(sev_counts, x="Severity", y="Count", color="Severity", color_discrete_map=color_map, template="plotly_dark")
        fig_sev.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_sev, use_container_width=True)

    with col_chart2:
        st.subheader("Risk Score Distribution")
        if "risk_score" in df_inc.columns and not df_inc["risk_score"].dropna().empty:
            fig_risk = px.histogram(df_inc, x="risk_score", nbins=20, template="plotly_dark", color_discrete_sequence=["#ff4b4b"])
            fig_risk.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_risk, use_container_width=True)

    # Incidents List Table
    st.subheader("Detailed Incident Stream")
    cols_to_show = [c for c in [
        "id", "timestamp", "attack_type", "severity", "detection_method",
        "model_confidence", "anomaly_score", "risk_score", "src_ip", "dst_ip", "status"
    ] if c in df_inc.columns]

    st.dataframe(df_inc[cols_to_show], use_container_width=True, hide_index=True)

    # Fast action banner
    st.markdown("---")
    st.markdown("💡 **Tip:** Go to **Incident Investigation** page to see deep SHAP attribution values and full flow feature metadata.")
