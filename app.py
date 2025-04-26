import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Used Car Price Predictor", layout="wide")

# App title and description
st.title("🚗Used Car Price Prediction App")
st.markdown("""
This application predicts the price of used cars based on different features:
- Car Brand/Name
- Car Age
- Mileage
""")

# Load used car dataset
@st.cache_data
def load_used_car_dataset():
    df = pd.read_csv('data/used_cars_data.csv')
    
    # Clean Mileage column (keep only numeric value)
    df['Mileage'] = df['Mileage'].str.extract(r'(\d+\.\d+)').astype(float)

    # Create Age feature
    df['Age'] = 2025 - df['Year']

    # Encode Name
    encoder = LabelEncoder()
    df['Name_encoded'] = encoder.fit_transform(df['Name'])
    
    # Select relevant features
    features = df[['Name', 'Name_encoded', 'Age', 'Mileage', 'Price']].dropna()

    return features, encoder

try:
    # Load dataset
    df, encoder = load_used_car_dataset()
    st.success("Dataset loaded successfully!")
    
    # Display dataset sample
    st.subheader("Dataset Sample")
    st.dataframe(df.head())
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Data Exploration", "Individual Feature Analysis", "Price Prediction"])
    
    with tab1:
        st.header("Data Exploration")
        
        # Dataset statistics
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe())
        
        # Correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df[['Name_encoded', 'Age', 'Mileage', 'Price']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        plt.title('Correlation Matrix of Used Car Features')
        st.pyplot(fig)
        
        # Correlation with price
        st.subheader("Correlation with Price")
        st.write(correlation_matrix['Price'].sort_values(ascending=False))
    
    with tab2:
        st.header("Individual Feature Analysis")
        
        # Feature selection
        feature = st.selectbox(
            "Select a feature to analyze its relationship with price:",
            options=['Name_encoded', 'Age', 'Mileage']
        )
        
        # Linear regression for selected feature
        X = df[[feature]].values
        y = df['Price'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Display regression equation and R²
        st.subheader(f"Linear Regression with {feature}")
        st.write(f"**Equation:** Price = {slope:.2f} × {feature} + {intercept:.2f}")
        st.write(f"**R² score:** {r2:.4f}")
        
        # Visualization of the relationship
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X, y, color='blue', label='Cars')
        ax.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
        
        # Add equation and R² on plot
        equation = f'Price = {slope:.2f} × {feature} + {intercept:.2f}'
        r2_text = f'R² = {r2:.4f}'
        ax.annotate(equation + '\n' + r2_text,
                   xy=(0.05, 0.95),
                   xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
        
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Car Price (₹)', fontsize=12)
        ax.set_title(f'Car Price vs {feature}', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
    
    with tab3:
        st.header("Car Price Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Single Feature Prediction")
            
            # Select feature for prediction
            single_feature = st.selectbox(
                "Select a feature for prediction:",
                options=['Name_encoded', 'Age', 'Mileage']
            )
            
            # Input value
            if single_feature == 'Name_encoded':
                car_name = st.selectbox("Select Car Brand:", options=sorted(df['Name'].unique()))
                feature_value = df[df['Name'] == car_name]['Name_encoded'].values[0]
                st.info(f"Selected car brand encoded as: {feature_value}")
            elif single_feature == 'Age':
                feature_value = st.slider("Car Age (years):", min_value=1, max_value=20, value=5)
            else:  # Mileage
                feature_value = st.slider("Car Mileage (kmpl):", min_value=10.0, max_value=30.0, value=20.0, step=0.5)
            
            # Build and use the model
            X_single = df[[single_feature]].values
            y_single = df['Price'].values
            
            single_model = LinearRegression()
            single_model.fit(X_single, y_single)
            
            single_predicted_price = single_model.predict([[feature_value]])[0]
            
            st.metric("Predicted Price (₹)", f"{single_predicted_price:.2f} lakhs")
        
        with col2:
            st.subheader("Multiple Feature Prediction")
            
            # Select car brand
            car_name_multi = st.selectbox("Select Car Brand:", options=sorted(df['Name'].unique()), key="multi_name")
            name_encoded_value = df[df['Name'] == car_name_multi]['Name_encoded'].values[0]
            
            # Input other features
            age_value = st.slider("Car Age (years):", min_value=1, max_value=20, value=5, key="multi_age")
            mileage_value = st.slider("Car Mileage (kmpl):", min_value=10.0, max_value=30.0, value=20.0, step=0.5, key="multi_mileage")
            
            # Build and use the model
            X_multi = df[['Name_encoded', 'Age', 'Mileage']].values
            y_multi = df['Price'].values
            
            multi_model = LinearRegression()
            multi_model.fit(X_multi, y_multi)
            
            new_car = np.array([[name_encoded_value, age_value, mileage_value]])
            multi_predicted_price = multi_model.predict(new_car)[0]
            
            st.metric("Predicted Price (₹)", f"{multi_predicted_price:.2f} lakhs")
            
            # Feature importance
            st.subheader("Feature Importance")
            coef_df = pd.DataFrame({
                'Feature': ['Brand', 'Age', 'Mileage'],
                'Coefficient': multi_model.coef_
            })
            coef_df['Absolute Coefficient'] = np.abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values('Absolute Coefficient', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Absolute Coefficient', y='Feature', data=coef_df, ax=ax)
            ax.set_title('Feature Importance (Absolute Coefficient Values)')
            st.pyplot(fig)
        
        # Model formula
        st.subheader("Multiple Regression Model Formula")
        st.write(f"Price = {multi_model.coef_[0]:.2f} × Brand + {multi_model.coef_[1]:.2f} × Age + {multi_model.coef_[2]:.2f} × Mileage + {multi_model.intercept_:.2f}")
        
        # Model performance
        st.subheader("Model Performance")
        y_multi_pred = multi_model.predict(X_multi)
        multi_r2 = r2_score(y_multi, y_multi_pred)
        st.write(f"R² score: {multi_r2:.4f}")
        
        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_multi, y_multi_pred, color='blue', alpha=0.5)
        ax.plot([min(y_multi), max(y_multi)], [min(y_multi), max(y_multi)],
                color='red', linestyle='--')
        ax.set_xlabel('Actual Price (₹ in lakhs)', fontsize=12)
        ax.set_ylabel('Predicted Price (₹ in lakhs)', fontsize=12)
        ax.set_title('Multiple Linear Regression: Actual vs Predicted Car Prices', fontsize=14)
        ax.grid(True)
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error loading the dataset: {e}")
    st.info("Make sure the file 'data/used_cars_data.csv' exists in the correct location.")
    st.code("""
    Expected file structure:
    - app.py (this file)
    - data/
        - used_cars_data.csv
    """)

# Add app footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This app uses linear regression models to predict used car prices based on historical data.
The predictions are based on brand reputation (encoded name), car age, and mileage.
""")