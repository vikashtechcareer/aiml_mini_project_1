import streamlit as st
import pandas as pd
import os
import plotly.express as px


def load_logs():
    logs = []

    for version in ['v1', 'v2']:
        log_path = f'/logs/model_{version}.log'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    try:
                        logs.append(eval(line.strip()))
                    except:
                        pass

    return pd.DataFrame(logs)

st.set_page_config(page_title="A/B Testing Dashboard", layout="wide")
st.title("🔍 Titanic A/B Model Testing Dashboard")

df = load_logs()

if not df.empty:
    st.metric("Total Requests", len(df))
    model_counts = df['model_version'].value_counts().reset_index()
    model_counts.columns = ['model_version', 'count']

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🔢 Request Count (Bar Chart)")
        st.bar_chart(model_counts.set_index('model_version'))

    with col2:
        st.write("### 🥧 Traffic Share (Pie Chart)")
        fig = px.pie(model_counts, names='model_version', values='count',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title='Request Distribution')
        st.plotly_chart(fig, use_container_width=True)
    st.write("## Recent Predictions")
    st.dataframe(df.sort_values(by="timestamp", ascending=False).head(20))
else:
    st.warning("No predictions yet. Make some requests!")
