

# 🚗 Used Car Price Prediction App

This is a **Streamlit-based web application** that predicts the price of used cars based on their brand, age, and mileage.  
The app uses **Linear Regression** and **Multiple Linear Regression** models trained on real-world data sourced from **carfores.com**.

---

## 📚 Project Overview

**Aim:**  
To build an interactive app where users can explore used car data, analyze features, and predict car prices using machine learning models.

**Technologies Used:**
- Python
- Streamlit
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn

**Machine Learning Models:**
- Simple Linear Regression
- Multiple Linear Regression

---

## ⚙️ Features

- **Dataset Exploration**:  
  View and understand the statistics and correlations between car features.
  
- **Individual Feature Analysis**:  
  Explore how each feature (Brand, Age, Mileage) affects the car price with regression plots.

- **Price Prediction**:
  - **Single Feature Prediction**: Predict car prices based on one selected feature.
  - **Multiple Feature Prediction**: Predict car prices using all features combined for better accuracy.

- **Model Insights**:
  - Display regression equations and R² scores.
  - Visualize feature importance.
  - Compare actual vs predicted prices.

- **Error Handling**:
  - Handles missing data.
  - Informs user if dataset is not found.

---

## 🛠️ How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/used-car-price-predictor.git
   cd used-car-price-predictor
   ```

2. **Install required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the folder structure**:
   ```
   - app.py (this Streamlit app)
   - data/
       - used_cars_data.csv
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure

```
├── app.py
├── data/
│   └── used_cars_data.csv
├── README.md
└── requirements.txt
```

---

## 📊 Screenshots

"C:\Users\Raza\OneDrive\图片\Screenshots\Screenshot 2025-04-26 160718.png"


---

## 📈 Model Details

- **Single Feature Model**:
  - Linear regression between one feature and price.
  - R² score displayed to show model accuracy.

- **Multiple Feature Model**:
  - Regression using all three features.
  - Higher R² score indicating better predictions.
  - Display of model formula:
    ```
    Price = (coef1 × Brand) + (coef2 × Age) + (coef3 × Mileage) + Intercept
    ```

---

## 🚀 Future Improvements

- Use more advanced regression models like Ridge, Lasso, or XGBoost.
- Add more features (e.g., car engine size, transmission type).
- Improve dataset cleaning and preprocessing.
- Deploy the app online (e.g., using Streamlit Cloud or AWS).

---

## 🧠 Author

- **Your Name** – [GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-profile)

---

## 📜 License

This project is licensed under the **MIT License**.

---

# ⭐ Special Note

This project was built as part of an **AI & ML** course assignment focused on applying **Linear and Multiple Linear Regression** in real-world datasets.

---

Would you also like me to create a sample `requirements.txt` file for you? 🚀  
It'll make running your project even easier! 🌟
