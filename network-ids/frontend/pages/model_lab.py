"""
PAGE 5: Model Laboratory
Dataset management, model training, performance comparison, confusion matrices, and model registry.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from frontend.api_client import APIClient


def render(api: APIClient):
    st.markdown("## 🧪 Machine Learning Model Laboratory")
    st.markdown("Train, evaluate, benchmark, and deploy supervised detection algorithms (XGBoost, Random Forest, Logistic Regression, LightGBM).")

    tab1, tab2, tab3 = st.tabs(["📁 Datasets & Validation", "⚡ Train & Compare", "📊 Evaluation Reports"])

    # TAB 1: Datasets
    with tab1:
        st.subheader("Import & Validate Datasets")
        up_file = st.file_uploader("Upload Network Dataset (CSV)", type=["csv"])
        if up_file is not None:
            if st.button("Upload to Server"):
                with st.spinner("Uploading and analyzing..."):
                    res = api.upload_dataset(up_file.name, up_file.getvalue())
                    if res:
                        st.success(f"Uploaded {up_file.name} successfully ({res.get('rows')} rows, {res.get('columns')} columns).")
                    else:
                        st.error("Failed to upload dataset.")

        st.markdown("---")
        st.subheader("Available Datasets")
        datasets = api.get_datasets() or []
        if datasets:
            df_ds = pd.DataFrame(datasets)
            st.dataframe(df_ds[["name", "type", "size_bytes", "path"]], use_container_width=True, hide_index=True)

            selected_ds = st.selectbox("Select Dataset to Validate", [d["name"] for d in datasets])
            if st.button("Validate Dataset"):
                val_res = api.validate_dataset(selected_ds)
                if val_res:
                    st.write("### Validation Summary")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rows", val_res.get("rows"))
                    c2.metric("Columns", val_res.get("columns"))
                    c3.metric("Duplicate Rows", val_res.get("duplicate_rows"))

                    if val_res.get("warnings"):
                        for w in val_res["warnings"]:
                            st.warning(w)

                    # Class Distribution
                    class_dist = val_res.get("class_distribution", {})
                    if class_dist:
                        st.write("#### Class Distribution")
                        df_cls = pd.DataFrame(list(class_dist.items()), columns=["Class", "Count"])
                        fig_cls = px.bar(df_cls, x="Class", y="Count", color="Class", template="plotly_dark")
                        st.plotly_chart(fig_cls, use_container_width=True)
        else:
            st.info("No datasets uploaded yet. Upload a CSV file (e.g. CIC-IDS2017 or UNSW-NB15) or generate test synthetic data.")

    # TAB 2: Model Training
    with tab2:
        st.subheader("Train New Model")
        c1, c2, c3 = st.columns(3)
        with c1:
            m_type = st.selectbox("Model Architecture", ["xgboost", "random_forest", "logistic_regression", "lightgbm"])
        with c2:
            m_task = st.selectbox("Classification Task", ["binary", "multiclass"])
        with c3:
            ds_names = [d["name"] for d in datasets] if datasets else ["synthetic_dataset"]
            m_ds = st.selectbox("Training Dataset", ds_names)

        if st.button("🚀 Start Training", type="primary"):
            with st.spinner(f"Training {m_type} on {m_ds} with SMOTE oversampling..."):
                t_res = api.train_model(m_type, m_task, m_ds)
                if t_res:
                    st.success(f"Model `{t_res.get('name')}` trained and deployed successfully!")
                else:
                    st.error("Training failed. Ensure dataset has clean features and labels.")

        st.markdown("---")
        st.subheader("Model Registry")
        models = api.get_models() or []
        if models:
            df_mods = pd.DataFrame(models)
            st.dataframe(df_mods[["name", "model_type", "task", "dataset", "is_active", "training_samples", "created_at"]], use_container_width=True, hide_index=True)

            # Activate model
            mod_names = [m["name"] for m in models]
            to_activate = st.selectbox("Set Active Model for Live Inference", mod_names)
            if st.button("Set Active"):
                res_act = api.activate_model(to_activate)
                if res_act:
                    st.success(f"Model `{to_activate}` is now actively serving detection engines.")
        else:
            st.info("No trained models in registry yet.")

    # TAB 3: Evaluation Reports
    with tab3:
        st.subheader("Evaluate Model Performance")
        if models and datasets:
            e1, e2 = st.columns(2)
            with e1:
                eval_mod = st.selectbox("Select Model to Evaluate", [m["name"] for m in models], key="eval_mod_select")
            with e2:
                eval_ds = st.selectbox("Evaluation Dataset", [d["name"] for d in datasets], key="eval_ds_select")

            if st.button("Run Evaluation"):
                with st.spinner("Running inference and computing performance metrics..."):
                    eval_res = api.evaluate_model(eval_mod, eval_ds)
                    if eval_res:
                        st.success("Evaluation complete!")
                        # Metrics Cards
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Accuracy", f"{eval_res.get('accuracy', 0):.2%}")
                        m2.metric("Precision (Macro)", f"{eval_res.get('precision_macro', 0):.2%}")
                        m3.metric("Recall (Macro)", f"{eval_res.get('recall_macro', 0):.2%}")
                        m4.metric("F1-Score (Macro)", f"{eval_res.get('f1_macro', 0):.2%}")
                        m5.metric("Latency", f"{eval_res.get('inference_latency_ms', 0):.2f} ms")

                        # Confusion Matrix
                        cm = eval_res.get("confusion_matrix")
                        if cm and len(cm) > 0:
                            st.write("#### Confusion Matrix")
                            z = cm
                            x = [f"Pred {i}" for i in range(len(cm[0]))]
                            y = [f"True {i}" for i in range(len(cm))]
                            fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale="Blues")
                            fig_cm.update_layout(template="plotly_dark", height=400)
                            st.plotly_chart(fig_cm, use_container_width=True)
                    else:
                        st.error("Evaluation failed.")
        else:
            st.info("Upload datasets and train models first to view evaluation reports.")
