"""
PAGE 6: Mitigation Centre & Firewall Management
Controlled automated mitigation, IP blocking/unblocking, and nftables rule audit trail.
"""
import streamlit as st
import pandas as pd
from frontend.api_client import APIClient


def render(api: APIClient):
    st.markdown("## 🛡️ Mitigation Centre & Firewall Rules")
    st.markdown("Automated and administrator-supervised IP containment using Linux nftables with full auditability.")

    tab1, tab2 = st.tabs(["Active Blocks", "Mitigation History & Audits"])

    with tab1:
        st.subheader("Manual IP Containment")
        c1, c2, c3 = st.columns(3)
        with c1:
            target_ip = st.text_input("Target IP Address", placeholder="e.g. 192.168.1.105")
        with c2:
            reason = st.text_input("Block Reason", placeholder="e.g. Repeated brute force attack")
        with c3:
            duration = st.number_input("Duration (minutes)", min_value=1, max_value=1440, value=60)

        if st.button("🚫 Apply Firewall Block", type="primary"):
            if not target_ip:
                st.warning("Please enter a valid IP address.")
            else:
                with st.spinner(f"Applying rule for {target_ip}..."):
                    res = api.block_ip(target_ip, reason, duration_minutes=duration)
                    if res:
                        st.success(f"Successfully enacted containment rule for `{target_ip}` (Simulated: {res.get('is_simulated')}).")
                    else:
                        st.error("Failed to block IP. Check protected IP list and parameters.")

        st.markdown("---")
        st.subheader("Active Blocked IPs")
        active_data = api.get_active_blocks()
        if active_data and active_data.get("actions"):
            df_active = pd.DataFrame(active_data["actions"])
            st.dataframe(df_active[["target_ip", "reason", "duration_minutes", "expires_at", "is_simulated", "created_at"]], use_container_width=True, hide_index=True)

            st.write("### Unblock IP")
            unblock_ip_target = st.selectbox("Select IP to unblock", df_active["target_ip"].unique())
            if st.button("🔓 Lift Block"):
                res_un = api.unblock_ip(unblock_ip_target, reason="Admin manual unblock via dashboard")
                if res_un:
                    st.success(f"Rule for `{unblock_ip_target}` successfully removed.")
                    st.rerun()
                else:
                    st.error("Failed to unblock IP.")
        else:
            st.info("No active firewall blocks currently enforced.")

    with tab2:
        st.subheader("Historical Mitigation Audits")
        hist_data = api.get_mitigation_history()
        if hist_data and hist_data.get("actions"):
            df_hist = pd.DataFrame(hist_data["actions"])
            st.dataframe(df_hist[["id", "action_type", "target_ip", "reason", "is_simulated", "status", "created_at"]], use_container_width=True, hide_index=True)
        else:
            st.info("No historical mitigation actions recorded.")
