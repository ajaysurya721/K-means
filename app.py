import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="K-Means Clustering Project")

st.title("📊 K-Means Clustering (Browse Dataset)")

# ---------------------------------------
# Upload Dataset
# ---------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------------------
    # Select numeric columns
    # ---------------------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least two numeric columns.")
    else:
        st.subheader("Clustering Settings")

        col1 = st.selectbox("Select Feature 1", numeric_cols)
        col2 = st.selectbox("Select Feature 2", numeric_cols)

        k = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)

        # ---------------------------------------
        # Prepare data
        # ---------------------------------------
        X = df[[col1, col2]]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ---------------------------------------
        # Train KMeans
        # ---------------------------------------
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        df["Cluster"] = clusters

        # ---------------------------------------
        # Visualization
        # ---------------------------------------
        st.subheader("Cluster Visualization")

        fig, ax = plt.subplots()
        ax.scatter(df[col1], df[col2], c=df["Cluster"])
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)

        st.pyplot(fig)

        # ---------------------------------------
        # Show clustered data
        # ---------------------------------------
        st.subheader("Clustered Dataset")
        st.dataframe(df)
