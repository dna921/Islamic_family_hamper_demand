
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os
import re
import pickle
import shap
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline

# -------------------------------
# Area Mapping
# -------------------------------
broad_areas = [
    'Londonderry', 'NAIT/Kingway', 'Boyle Street/McCauley', 'Windermere', 'Castle Downs',
    'Northeast Edmonton', 'Clareview', 'Alberta Avenue', 'Mill Woods', 'Blue Quill',
    'Strathcona', 'Jasper Place', 'Malmo Plains', 'Bonnie Doon', 'Ellerslie',
    'West Edmonton', 'Calder/Kensington', 'Oliver/Downtown', 'Albany/Cumberland',
    'Beverly', 'North Central', 'University Area', 'Meadowlark/Jasper Place', 'Eastgate',
    'Capilano', 'The Hamptons', 'Terwillegar', 'Downtown', 'Red Deer', 'The Meadows', 'Camrose'
]
broad_area_mapping = {area: idx for idx, area in enumerate(broad_areas)}

# -------------------------------

# -------------------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# Load the dataset with a specified encoding

# Page 1: Dashboard
def dashboard():
   st.image('Islamic Family.webp', caption="Islamic Family", use_container_width=True)

      # Add logos on the right (col3)

      # Highlight the project title
   st.title("Islamic Family Hamper Distribution Dashboard")

    # Add three summary statistics cards
   st.write("### Project Highlights")
   col1, col2, col3 = st.columns(3)
   col1.metric(label="Food hampers distributed monthly", value="2000+")
   col2.metric(label="Total volunteers engaged", value="2100")
   col3.metric(label="Neighborhoods Covered", value="40+")

    # Use tabs to organize content
   tab1, tab2, tab3 = st.tabs(["Abstract", "What the Project Does", "Inspiration"])

    # Abstract Tab
   with tab1:
        st.write("💡 **Abstract**")
        st.info(
            """
            This project focuses on identifying high-demand neighborhoods in Edmonton for Islamic food hamper distribution. Using past distribution records
            and socio-economic data, we built a machine learning model to predict which areas need the most support. The process involved data cleaning, analysis,
             and training regression and classification models. Our goal is to help Islamic organizations allocate resources more effectively and equitably.
            """
        )

    # What the Project Does Tab
   with tab2:
        st.write("👨🏻‍💻 **What Our Project Does**")
        bullet_points = [
            "Predicts food hamper demand across Edmonton neighborhoods using machine learning.",
            "Identifies key socio-economic factors influencing high-need areas.",
            "Analyzes past distribution data to uncover patterns and trends.",
            "Visualizes geographic and temporal trends in hamper distribution.",
            "Enhances planning for future food drives with data-driven insights.",
        ]
        for point in bullet_points:
            st.markdown(f"- {point}")

    # Inspiration Tab
   with tab3:
        st.write("🌟 **Inspiration**")
        st.warning(
            """
            The inspiration for this project came from the Islamic community’s ongoing efforts to support vulnerable families
             in Edmonton through food drives. While the intention and effort are strong, many of these initiatives face challenges
              in identifying which neighborhoods need the most support. By combining data science with community service, we aim to
               help Islamic organizations distribute food more effectively, ensuring that help reaches those who need it the most. Our goal is to turn compassion into targeted action through the power of data..
            """
        )
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st






def explainable_ai_page():
    st.title("🧠 Explainable AI Insights")
    st.write("### Understanding What Drives Food Hamper Needs")

    # Load model
    booster = xgb.Booster()
    booster.load_model("xgb_model.json")

    # Load data
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv")

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    # Predict using Booster and DMatrix
    y_pred = booster.predict(xgb.DMatrix(X_test))

    # If output is 2D (multi-output), use only column 0
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 0]

    # Now calculate residuals
    residuals = np.ravel(y_test) - np.ravel(y_pred)

    # Residuals
    y_pred = np.ravel(y_pred)
    y_test = np.ravel(y_test.values)
    residuals = y_test - y_pred

    # 📉 Residual Plot
    st.subheader("📉 Residual Plot")
    st.markdown("This plot shows how accurate our model is. The closer the dots are to the red line, the better the predictions.")

    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuals, ax=ax1)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_xlabel("Predicted Hamper Demand")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residual Plot for Hamper Demand")
    st.pyplot(fig1)


    # 🧠 SHAP Summary Plot
    st.subheader("📌 SHAP Summary Plot")
    

    # Create SHAP explainer
    explainer = shap.Explainer(booster)
    shap_values = explainer(X_test)

    # Plot to matplotlib figure and display in Streamlit
    st.subheader("SHAP Summary Plot - Hamper Demand")
    st.markdown("""
🧠 **Interpretation Tip**  
Each dot represents one prediction made by the model.  

- **Position**: The further a dot is from the center (left or right), the more impact that feature had on the prediction.
- **Color**: Indicates the feature value.  
   - **Red = high value**  
   - **Blue = low value**

📌 For example, more **dependents** (red dots on the right) tend to **increase** predicted hamper demand.  
Lower **distance** (blue dots on the left) may **decrease** the prediction slightly.

This helps us see which features matter the most when estimating food hamper needs.
""")

    fig_summary, ax_summary = plt.subplots()
    shap.summary_plot(shap_values.values, X_test, show=False)
    st.pyplot(fig_summary)
    

        # Only keep first target's SHAP values (hamper_demand)
    X_test_shap = pd.read_csv("X_test.csv")  # ← this loads your SHAP input
    shap_values = explainer(X_test_shap)

    shap_vals_hamper = shap_values.values  # ✅ correct for regression

    # Check shape consistency
    if shap_vals_hamper.shape[0] != X_test_shap.shape[0]:
        st.error("Mismatch in SHAP values and test data samples!")
    else:
        # Compute mean importance
        shap_importance = np.abs(shap_vals_hamper).mean(axis=0)
        top_indices = np.argsort(shap_importance)[-5:][::-1]

        top_features = [X_test_shap.columns[i] for i in top_indices]

        st.subheader("Dependence Plots (Top 5)")
        st.markdown("""
📊 **Dependence Plot Guide**

These plots show how each top feature affects the model’s prediction, one at a time.

- **X-axis**: The actual value of the feature (e.g., number of dependents).
- **Y-axis**: The SHAP value (how much that feature pushed the prediction up or down).
- **Color**: Shows interaction with another feature (e.g., distance or dependents).

### How to read this:
- If the dots go **upward**, higher values of the feature **increase** the predicted hamper demand.
- If they go **downward**, higher values **decrease** the prediction.

📌 Example:  
In the `dependents_qty` plot, as the number of dependents increases, the model tends to predict **more** food hampers needed.

These insights help us understand what factors influence need in each neighborhood.
""")

    for feature in top_features:
      st.write(f"SHAP Dependence Plot: {feature}")
      shap.dependence_plot(feature, shap_vals_hamper, X_test_shap, show=False)
      fig = plt.gcf()
      st.pyplot(fig)
      plt.clf()




    st.success("SHAP-based insights generated using the XGBoost `.json` model.")
# Page 3: Machine Learning Modeling
import streamlit as st
import pandas as pd
import pickle

def machine_learning_modeling():
    st.title("📦 Hamper Demand Prediction")
    st.write("Fill in the household and request details below:")

    try:
        # 📥 User Inputs
        dependents = st.number_input("👨‍👩‍👧 Number of Dependents", min_value=0)
        distance = st.number_input("📍 Distance (km) from center", min_value=0.0)

        # Month with names
        month_dict = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        month_name = st.selectbox("📆 Month", list(month_dict.keys()))
        month = month_dict[month_name]

        requests = st.number_input("📝 Requests per Household", min_value=0.0)
        avg_size = st.number_input("👥 Average Household Size", min_value=0.0)

        # 📍 Broad Area selection
        broad_areas = [
            'Londonderry', 'NAIT/Kingway', 'Boyle Street/McCauley', 'Windermere', 'Castle Downs',
            'Northeast Edmonton', 'Clareview', 'Alberta Avenue', 'Mill Woods', 'Blue Quill',
            'Strathcona', 'Jasper Place', 'Malmo Plains', 'Bonnie Doon', 'Ellerslie',
            'West Edmonton', 'Calder/Kensington', 'Oliver/Downtown', 'Albany/Cumberland',
            'Beverly', 'North Central', 'University Area', 'Meadowlark/Jasper Place', 'Eastgate',
            'Capilano', 'The Hamptons', 'Terwillegar', 'Downtown', 'Red Deer', 'The Meadows', 'Camrose'
        ]
        broad_area_mapping = {area: idx for idx, area in enumerate(broad_areas)}
        broad_area = st.selectbox("🏘️ Broad Area", broad_areas)
        broad_area_encoded = broad_area_mapping[broad_area]

        # 👻 Socio-economic defaults
        input_data = pd.DataFrame([{
            'dependents_qty': dependents,
            'primary_client_key': 0,
            'house_number': 0,
            'distance_km': distance,
            'month': month,
            'requests_per_household': requests,
            'week_of_month': 1,  # 👈 Defaulted internally
            'Average household size': avg_size,
            'broad_area_encoded': broad_area_encoded,
            'Couple-family households with children': 5265.0,
            'Multigenerational households': 195,
            'One-parent families in which the parent is a man+': 1950,
            'One-parent families in which the parent is a woman+': 2535,
            'Persons not in census families - Living alone': 1468,
            'Total one-parent families': 0
        }])

        expected_columns = [
            'dependents_qty', 'primary_client_key', 'house_number', 'distance_km', 'month',
            'requests_per_household', 'week_of_month', 'Average household size', 'broad_area_encoded',
            'Couple-family households with children', 'Multigenerational households',
            'One-parent families in which the parent is a man+',
            'One-parent families in which the parent is a woman+',
            'Persons not in census families - Living alone', 'Total one-parent families'
        ]

        input_data = input_data[expected_columns]

        # 🔍 Load Trained Model
        try:
            with open("xgb_model.pkl", "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            return

        # 🔮 Make Prediction
        hamper_pred = model.predict(input_data.values)[0]
        max_hamper_value = 100
        hamper_pred = hamper_pred * max_hamper_value

        st.success(f"📦 Predicted Hamper Demand: **{int(round(hamper_pred))} hampers**")

        # 📊 Hamper Demand by Week from historical data
        st.markdown("### 📊 Hamper Demand by Week in Your Selected Area")

        try:
            df = pd.read_csv("df_merged_backup.csv")
            df['broad_area_clean'] = df['broad_area'].astype(str).str.strip().str.title()

            filtered_df = df[(df['broad_area_clean'] == broad_area) & (df['month'] == month)]

            if not filtered_df.empty and 'week_of_month' in filtered_df.columns:
                fig_week = px.bar(filtered_df, x='week_of_month', y='hamper_demand',
                                  title=f"Hamper Demand by Week - {broad_area} ({month_name})",
                                  labels={'week_of_month': 'Week of Month', 'hamper_demand': 'Hamper Demand'})
                st.plotly_chart(fig_week)
            else:
                st.info("No weekly data available for this area and month.")
        except Exception as e:
            st.error(f"Error loading weekly hamper demand chart: {e}")

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")

# Page 4: Neighbourhood Mapping

# Read geospatial data

import streamlit as st
import pandas as pd
import plotly.express as px

def hamper_demand_map():
    st.title("📍 Hamper Demand Map by Broad Area")

    # Load main data and combined coordinates
    df = pd.read_csv("df_merged_backup.csv")
    coords = pd.read_csv("geocoded_broad_areas.csv")
    manual_coords = pd.read_csv("location.csv")

    # Clean and combine both coordinate sources
    coords['broad_area'] = coords['broad_area'].astype(str).str.strip().str.title()
    manual_coords = manual_coords.rename(columns={"Neighbourhood": "broad_area"})
    manual_coords['broad_area'] = manual_coords['broad_area'].astype(str).str.strip().str.title()

    # Combine both sources
    all_coords = pd.concat([manual_coords[['broad_area', 'Latitude', 'Longitude']],
                            coords[['broad_area', 'Latitude', 'Longitude']]])
    all_coords = all_coords.drop_duplicates(subset='broad_area', keep='first')

    # Prepare df and merge with coordinates
    df['broad_area_clean'] = df['broad_area'].astype(str).str.strip().str.title()
    df = pd.merge(df, all_coords, left_on='broad_area_clean', right_on='broad_area', how='left')
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Group data for plotting
    summary = df.groupby('broad_area_clean', as_index=False).agg({
        'hamper_demand': lambda x: int(round(x.mean())),
        'Latitude': 'first',
        'Longitude': 'first'
    })

    # Plot
    fig = px.scatter_mapbox(
        summary,
        lat='Latitude',
        lon='Longitude',
        color='hamper_demand',
        hover_name='broad_area_clean',
        hover_data={'hamper_demand': True},
        zoom=10,
        title="Hamper Demand by Broad Area"
    )
    fig.update_layout(mapbox_style='open-street-map')
    st.plotly_chart(fig)

def interactive_eda_page():
    st.title("📊 Interactive EDA - Hamper Demand Insights")
    st.write("Use the filters below to explore hamper demand and volunteer data across neighborhoods.")

    # Load data
    df = pd.read_csv("df_merged_backup.csv")

    # Clean area names
    df['broad_area_clean'] = df['broad_area'].astype(str).str.strip().str.title()

    # Filter widgets
    area_list = sorted(df['broad_area_clean'].dropna().unique())
    selected_area = st.selectbox("🏘️ Select Broad Area", area_list)
    month_dict = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
      }
    month_name = st.selectbox("📅 Select Month", list(month_dict.keys()))
    selected_month = month_dict[month_name]

    # Filtered DataFrame
    filtered_df = df[(df['broad_area_clean'] == selected_area) & (df['month'] == selected_month)]

    st.markdown(f"### 📌 Overview for **{selected_area}**")

    # 📦 Hamper Demand by Week of Month
    if 'week_of_month' in filtered_df.columns:
        fig1 = px.bar(filtered_df, x='week_of_month', y='hamper_demand',
                      title="📦 Hamper Demand by Week",
                      labels={'week_of_month': 'Week of Month', 'hamper_demand': 'Hamper Demand'})
        st.plotly_chart(fig1)

        # 📈 Hamper Demand by Month (Interactive Bar Chart)
    st.markdown("### 📦 Monthly Hamper Demand Overview")

    # Ensure necessary columns exist
    if 'month' in df.columns and 'hamper_demand' in df.columns:
        # Map month number to name
        month_dict = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }

        # Group by month and sum hamper demand
        month_demand = df.groupby('month')['hamper_demand'].sum().reset_index()
        month_demand['Month Name'] = month_demand['month'].astype(int).map(month_dict)

        # Sort by calendar month
        month_demand = month_demand.sort_values('month')

        # Plot
        fig_month = px.bar(
            month_demand,
            x='Month Name',
            y='hamper_demand',
            title="📅 Total Hamper Demand by Month",
            labels={'hamper_demand': 'Total Hamper Demand', 'Month Name': 'Month'},
            color='hamper_demand',
            color_continuous_scale='Blues'
        )

        st.plotly_chart(fig_month)


    # 👨‍👩‍👧 Dependents Distribution
    if 'dependents_qty' in filtered_df.columns:
        fig2 = px.histogram(filtered_df, x='dependents_qty', nbins=10,
                            title="👨‍👩‍👧 Dependents Distribution",
                            labels={'dependents_qty': 'Number of Dependents'})
        st.plotly_chart(fig2)

    # 📍 Distance from Center
    if 'distance_km' in filtered_df.columns:
        fig3 = px.histogram(filtered_df, x='distance_km', nbins=10,
                            title="📍 Distance from Center",
                            labels={'distance_km': 'Distance (km)'})
        st.plotly_chart(fig3)

    # 👥 Average Household Size across Areas (New Plot)
    if 'Average household size' in df.columns:
        avg_household = df.groupby('broad_area_clean')['Average household size'].mean().reset_index()
        fig4 = px.bar(avg_household, x='broad_area_clean', y='Average household size',
                      title='👥 Average Household Size by Area',
                      labels={'broad_area_clean': 'Broad Area', 'Average household size': 'Avg. Household Size'})
        fig4.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig4)

    # 📊 Socio-Economic Factors Across All Areas
    # 📊 Overall Socio-Economic Household Type Distribution (Pie Chart)
    st.markdown("### 🏘️ Overall Socio-Economic Household Composition")

    # List of relevant columns
    socio_cols = [
        'Couple-family households with children',
        'Multigenerational households',
        'One-parent families in which the parent is a man+',
        'One-parent families in which the parent is a woman+',
        'Persons not in census families - Living alone',
        'Total one-parent families'
    ]

    # Ensure all required columns exist
    if all(col in df.columns for col in socio_cols):
        # Aggregate total counts across all rows
        overall_household_summary = df[socio_cols].sum()

        # Plot as pie chart
        fig_pie = px.pie(
            names=overall_household_summary.index,
            values=overall_household_summary.values,
            title="Household Type Distribution Across All Areas",
            hole=0.4
        )
        st.plotly_chart(fig_pie)

    


# Page 5: Data Collection
def data_collection():
    st.title("Registration form")
    st.write("Please fill out the Google form to register to be a member of Islamic Family!")
    google_form_url = "https://docs.google.com/forms/d/e/1FAIpQLSdAczaPFRHCfVeq-qOVkHwn9QE3aCR5I-1xkJhV0XImfWF9Xg/viewform?usp=dialog"
    st.markdown(f"[Fill out the form]({google_form_url})")

def thank_you_page():
    st.title("Thank You to Our Stakeholders!")

    # Add a thank you message
    st.write(
        """
        We extend our heartfelt gratitude to our amazing stakeholders for their invaluable support in making this project a success.
        Your contributions and dedication have made a meaningful impact on our community.
        """
    )

    # Create a layout with columns for displaying logos
    st.write("### Our Stakeholders:")
    st.image('67364fb3e83436f479ca372a_IF-Homepage-Hero.webp', caption="The Islamic Family", use_container_width=True)


    # Add a footer with a final thank you message
    st.write("---")
    st.subheader("Together, we are making a difference!")



# -------------------------------------
def main():
    st.sidebar.title("Islamic Family App")
    app_page = st.sidebar.radio("Select a Page", [
        "Dashboard",
        "Interactive EDA",
        "Neighbourhood Mapping",
        "ML Modeling",
        "Explainable AI",
        "Data Collection",
        "Thank You"
    ])

    
    if app_page == "Dashboard":
        dashboard()  # Call your dashboard function
    elif app_page == "Interactive EDA":
        interactive_eda_page()
    elif app_page == "Neighbourhood Mapping":
        hamper_demand_map()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Explainable AI":
        explainable_ai_page()
    elif app_page == "Data Collection":
        data_collection()
    elif app_page == "Thank You":
        thank_you_page()

if __name__ == "__main__":
    main()
