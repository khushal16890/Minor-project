"""
PAGE 2: Live Network Monitoring
Visualizes network traffic trends, protocol breakdowns, connection flows, and live PCAP analysis.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from frontend.api_client import APIClient


def render(api: APIClient):
    st.markdown("## 📡 Live Network Traffic Monitor")
    st.markdown("Real-time inspection of captured network flows, protocol behavior, and connection telemetry.")

    # Controls
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        protocol_filter = st.selectbox("Protocol Filter", ["All", "tcp", "udp", "icmp"])
    with c2:
        label_filter = st.selectbox("Traffic Type", ["All", "BENIGN", "Malicious"])
    with c3:
        page_size = st.selectbox("Rows per page", [25, 50, 100], index=1)
    with c4:
        if st.button("🔄 Refresh Flows"):
            st.rerun()

    proto_arg = None if protocol_filter == "All" else protocol_filter
    label_arg = None if label_filter == "All" else ("BENIGN" if label_filter == "BENIGN" else "malicious")

    traffic_data = api.get_traffic(page=1, page_size=page_size, protocol=proto_arg, label=label_arg)
    if not traffic_data or not traffic_data.get("flows"):
        st.info("No network flows captured yet in the database. Upload a PCAP file or import a dataset.")
        return

    flows = traffic_data["flows"]
    df_flows = pd.DataFrame(flows)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flows Displayed", len(df_flows))
    total_bytes = df_flows["total_bytes"].fillna(0).sum() if "total_bytes" in df_flows.columns else 0
    m2.metric("Total Bytes", f"{total_bytes:,} B")
    unique_src = df_flows["src_ip"].nunique() if "src_ip" in df_flows.columns else 0
    m3.metric("Unique Sources", unique_src)
    unique_dst = df_flows["dst_ip"].nunique() if "dst_ip" in df_flows.columns else 0
    m4.metric("Unique Targets", unique_dst)

    # Charts Row
    ch1, ch2 = st.columns(2)
    with ch1:
        st.subheader("Protocol Distribution")
        if "protocol" in df_flows.columns and not df_flows["protocol"].dropna().empty:
            proto_counts = df_flows["protocol"].value_counts().reset_index()
            proto_counts.columns = ["Protocol", "Count"]
            fig_proto = px.pie(proto_counts, names="Protocol", values="Count", template="plotly_dark", hole=0.3)
            fig_proto.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_proto, use_container_width=True)

    with ch2:
        st.subheader("Flow Duration by Protocol")
        if "duration" in df_flows.columns and "protocol" in df_flows.columns:
            fig_dur = px.box(df_flows, x="protocol", y="duration", color="protocol", template="plotly_dark")
            fig_dur.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_dur, use_container_width=True)

    # Flows Table
    st.subheader("Captured Flows Telemetry")
    display_cols = [c for c in ["timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "duration", "total_bytes", "label", "source"] if c in df_flows.columns]
    st.dataframe(df_flows[display_cols], use_container_width=True, hide_index=True)
