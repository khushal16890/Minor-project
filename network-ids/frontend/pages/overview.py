"""
PAGE 1: Security Overview
Displays global cybersecurity metrics, threat distributions, and recent incidents.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from frontend.api_client import APIClient


def render(api: APIClient):
    st.markdown("## 🛡️ Security Operations Center Overview")
    st.markdown("Real-time monitoring and high-level posture assessment across all detection modules.")

    overview = api.get_dashboard_overview()
    if not overview:
        st.warning("⚠️ Backend API unavailable at http://127.0.0.1:8000. Please start the backend service.")
        return

    # Top Metric Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Flows", f"{overview.get('total_flows', 0):,}")
    c2.metric("Benign Flows", f"{overview.get('benign_flows', 0):,}")
    c3.metric("Malicious Flows", f"{overview.get('malicious_flows', 0):,}")
    c4.metric("Active Incidents", f"{overview.get('active_incidents', 0):,}")
    c5.metric("Critical Incidents", f"{overview.get('critical_incidents', 0):,}", delta_color="inverse")
    c6.metric("Active Blocks", f"{overview.get('active_blocks', 0):,}")

    st.markdown("---")

    # Visualizations Row
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Detected Attack Categories")
        attack_dist = overview.get("attack_distribution", {})
        if attack_dist:
            df_attacks = pd.DataFrame(
                list(attack_dist.items()), columns=["Attack Category", "Count"]
            ).sort_values(by="Count", ascending=False)
            fig_attacks = px.bar(
                df_attacks,
                x="Attack Category",
                y="Count",
                color="Count",
                color_continuous_scale="Reds",
                template="plotly_dark",
            )
            fig_attacks.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_attacks, use_container_width=True)
        else:
            st.info("No malicious attack categories detected yet.")

    with col_right:
        st.subheader("Detections by Engine")
        method_dist = overview.get("detection_method_counts", {})
        if method_dist:
            df_methods = pd.DataFrame(
                list(method_dist.items()), columns=["Engine", "Count"]
            )
            fig_methods = px.pie(
                df_methods,
                names="Engine",
                values="Count",
                hole=0.4,
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_methods.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_methods, use_container_width=True)
        else:
            st.info("No detection method statistics recorded yet.")

    # Recent Security Events
    st.subheader("Recent Security Events")
    events = overview.get("recent_events", [])
    if events:
        df_events = pd.DataFrame(events)
        # Format table
        st.dataframe(
            df_events[[
                "timestamp", "attack_type", "severity", "detection_method",
                "src_ip", "dst_ip", "risk_score", "status"
            ]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No recent security events logged.")
