
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.title("E-Commerce Customer Behavior Insights")

# Load data
df = pd.read_csv('../data/sample_data.csv')

st.write("Sample Data:", df.head())

# Dummy Feature Importances
features = ['time_spent', 'price', 'clicks']
importances = [0.4, 0.35, 0.25]
chart_data = pd.DataFrame({'Feature': features, 'Importance': importances}).set_index('Feature')

st.subheader("Feature Importances")
st.bar_chart(chart_data)
