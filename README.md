# Islamic Family Hamper Demand Prediction App

This web application was built as part of the CMPT 3835 project to help identify neighborhoods in Edmonton with the highest demand for food hampers. Using historical distribution data and socio-economic features, we trained a machine learning model to predict demand and explain the contributing factors.

## Project Objective

To assist Islamic organizations in better allocating food hampers by using data-driven predictions. The goal is to support families in need by identifying high-demand areas based on past trends and socio-economic indicators.

## Features

- Dashboard overview of the project
- Machine Learning model to predict hamper demand
- Interactive map showing demand by neighborhood
- SHAP-based explainable AI visualizations
- Registration page for new community members
- Stakeholder thank-you section

## Technologies Used

- Streamlit (frontend and app deployment)
- Python (pandas, numpy, matplotlib, seaborn)
- XGBoost (regression model)
- SHAP (model explainability)
- Plotly (interactive map)
- GitHub + Streamlit Cloud (deployment)

## File Structure

- `app.py`: Main application file with page navigation
- `prediction_page.py`, `explain_page.py`, `eda_page.py`: Functional pages
- `xgb_model.pkl` or `xgb_model.json`: Trained machine learning model
- CSV files: Input data for predictions and SHAP
- Image files: Visuals for EDA and thank-you sections


