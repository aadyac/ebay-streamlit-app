!pip install plotly  

import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="eBay Laptop Explorer", layout="wide")

# Title
st.title("💻 eBay Laptop Data Explorer & Price Predictor")

# Load the dataset
csv_path = "C:\\Users\\claud\\OneDrive\\Desktop\\messy\\Ebay Cleaned Data Set CSV.csv"
df = pd.read_csv(csv_path)
df = df.iloc[:, 1:]  # Remove the first column

# Preprocessing functions
def extract_numeric_speed(value):
    try:
        return float(value.split()[0])
    except (ValueError, AttributeError):
        return np.nan

def extract_numeric_ram(value):
    try:
        return float(value.split()[0])
    except (ValueError, AttributeError):
        return np.nan

def extract_numeric_ssd(value):
    try:
        numeric_value = ''.join(c for c in value if c.isdigit())
        return float(numeric_value) if numeric_value else np.nan
    except (ValueError, AttributeError):
        return np.nan

# Apply preprocessing
df['Processor Speed'] = df['Processor Speed'].apply(extract_numeric_speed)
df['Processor Speed'].fillna(df['Processor Speed'].mean(), inplace=True)

df['Ram Size'] = df['Ram Size'].apply(extract_numeric_ram)
df['Ram Size'].fillna(df['Ram Size'].mean(), inplace=True)

df['SSD Capacity'] = df['SSD Capacity'].apply(extract_numeric_ssd)
df['SSD Capacity'].fillna(df['SSD Capacity'].mean(), inplace=True)

# Show cleaned dataset
st.subheader("Cleaned Dataset")
st.dataframe(df)

# --- Filters ---
st.sidebar.header("🔎 Filter Options")

brands = df['Brand'].unique().tolist()
brand_filter = st.sidebar.multiselect("Select Brands", options=brands, default=brands)

min_price = float(df['Price'].min())
max_price = float(df['Price'].max())
price_range = st.sidebar.slider("Select Price Range", min_value=min_price, max_value=max_price,
                                value=(min_price, max_price))

conditions = df['Condition'].unique().tolist()
condition_filter = st.sidebar.multiselect("Select Conditions", options=conditions, default=conditions)

# Filter the dataset
filtered_data = df[
    (df['Brand'].isin(brand_filter)) &
    (df['Price'].between(price_range[0], price_range[1])) &
    (df['Condition'].isin(condition_filter))
]

# Show filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_data)

# --- Visualizations ---
st.subheader("📊 Visualizations")

# Average Price by Brand
avg_price_by_brand = filtered_data.groupby('Brand')['Price'].mean().sort_values(ascending=False)
fig_bar = px.bar(avg_price_by_brand, title='Average Price by Brand', labels={"value": "Average Price"})
st.plotly_chart(fig_bar, use_container_width=True)

# Box Plot
fig_box = px.box(filtered_data, x='Brand', y='Price', title='Price Range by Brand', color='Brand')
st.plotly_chart(fig_box, use_container_width=True)

# Scatter Plot
sort_by = st.selectbox("Scatter Plot X-axis", ['Price', 'Processor Speed'])
highlight_conditions = st.multiselect("Highlight Conditions in Scatter Plot", options=conditions, default=conditions)

scatter_data = filtered_data[filtered_data['Condition'].isin(highlight_conditions)]

fig_scatter = px.scatter(
    scatter_data,
    x=sort_by,
    y='Processor Speed',
    color='Brand',
    size='Price',
    title=f"{sort_by} vs Processor Speed"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# --- Price Prediction ---
st.subheader("🤖 Price Prediction")

with st.form("predict_form"):
    processor_speed = st.number_input("Processor Speed (GHz)", min_value=0.5, max_value=5.0, value=2.5)
    ram_size = st.number_input("RAM Size (GB)", min_value=2, max_value=128, value=8)
    ssd_capacity = st.number_input("SSD Capacity (GB)", min_value=32, max_value=2048, value=256)
    submitted = st.form_submit_button("Predict Price")

    if submitted:
        # Prepare data
        X = df[['Processor Speed', 'Ram Size', 'SSD Capacity']].fillna(0)
        y = df['Price']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make prediction
        input_features = scaler.transform([[processor_speed, ram_size, ssd_capacity]])
        predicted_price = model.predict(input_features)[0]
        st.success(f"💰 Predicted Price: **${predicted_price:,.2f}**")

