"""
Main Streamlit Dashboard Entry Point.
Intelligent Hybrid Network Intrusion Detection & Prevention System (NIDS/NIPS).
"""
import streamlit as st
from frontend.api_client import APIClient
from frontend.pages import (
    overview,
    live_monitoring,
    threat_detection,
    incident_investigation,
    model_lab,
    mitigation_centre,
    settings,
)

# Page configuration
st.set_page_config(
    page_title="Hybrid NIDS/NIPS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme CSS
st.markdown("""
<style>
    /* Dark Theme Customizations */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .metric-card {
        background-color: #1e222b;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #3b82f6;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API Client
api = APIClient()

# Sidebar Navigation
st.sidebar.title("🛡️ HYBRID NIDS/NIPS")
st.sidebar.caption("Intelligent Detection & Prevention Engine")

# System Status indicator
health = api.get_health()
if health and health.get("status") == "ok":
    st.sidebar.success(f"● System Online | Uptime: {health.get('uptime_seconds', 0):.0f}s")
else:
    st.sidebar.error("● Backend Offline (Port 8000)")

page_names = [
    "Security Overview",
    "Live Monitoring",
    "Threat Detection",
    "Incident Investigation",
    "Model Laboratory",
    "Mitigation Centre",
    "System Setting",
]

selected_page = st.sidebar.radio(
    "Navigation",
    page_names,
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("v1.0.0 | Defense-in-Depth Architecture")

# Route to selected page
if selected_page == "Security Overview":
    overview.render(api)
elif selected_page == "Live Monitoring":
    live_monitoring.render(api)
elif selected_page == "Threat Detection":
    threat_detection.render(api)
elif selected_page == "Incident Investigation":
    incident_investigation.render(api)
elif selected_page == "Model Laboratory":
    model_lab.render(api)
elif selected_page == "Mitigation Centre":
    mitigation_centre.render(api)
elif selected_page == "System Settings":
    settings.render(api)
