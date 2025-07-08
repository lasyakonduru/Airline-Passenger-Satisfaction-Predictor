import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Streamlit page config
st.set_page_config(page_title="Airline Satisfaction Predictor", layout="wide")

# Loading model & cleaned data
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_airline_passenger_satisfaction.csv")

@st.cache_resource
def load_model():
    return joblib.load("rf_airline_model.pkl")

df = load_data()
model = load_model()
feat_order = joblib.load("rf_model_features.pkl")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 EDA", "🔮 Predict"])

# 🏠 HOME PAGE
if page == "🏠 Home":
    # Main Title
    st.title("✈️ Welcome to the Airline Passenger Satisfaction App")
    
    # Create Tabs
    tab1, tab2 = st.tabs(["👋 Welcome", "📊 Data"])

    # TAB 1: WELCOME
    with tab1:
        st.subheader("👋 Welcome Aboard!")
        st.image("airline.jpg", use_container_width = True)
    
        st.markdown("""
        This app helps users explore factors that influence **airline passenger satisfaction** and provides **quick predictions** based on customer details.
    
        ### ✨ Purpose of This App
        - Help airlines understand customer satisfaction drivers  
        - Provide real-time predictions using machine learning  
        - Enable data-driven decision making in the aviation industry
    
        ### 👥 Who Can Use This App?
        - **Airline Customer Experience Teams**: Identify improvement areas  
        - **Data Analysts & Students**: Learn classification and data visualization  
        - **Travel Startups**: Integrate satisfaction predictors into recommendation systems  
    
        ---
        👉 Head over to the **Data** tab to explore the data and background.  
        """)

    # TAB 2: DATA
    with tab2:
        st.subheader("📄 Dataset Overview")
        st.markdown("This project uses the [Airline Passenger Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) from Kaggle.")

        st.markdown("### 🔹 First 5 Records")
        st.dataframe(df.head())

        st.markdown("### 🔹 Dataset Information")
        st.write(f"- Number of records: `{df.shape[0]}`")
        st.write(f"- Number of features: `{df.shape[1]}`")
    
        st.markdown("### 🔹 Target Variable")
        st.write("**`satisfaction`** – Binary outcome: `'satisfied'` or `'neutral/dissatisfied'`")
    
        st.markdown("### 🔹 Predictor Variables Include:")
        st.markdown("""
        - **Passenger Profile**: Age, Gender, Customer Type  
        - **Travel Info**: Class, Type of Travel, Flight Distance  
        - **Delay Info**: Departure Delay, Arrival Delay  
        - **Service Ratings**: Seat comfort, Online boarding, Cleanliness, Wifi, etc.
        """)
    
        st.markdown("---")
        st.subheader("🌲 Our Random Forest Classifier")
        st.write("We tested 3 models: **KNN**, **Logistic Regression**, and **Random Forest**. The Random Forest model stood out with the highest test accuracy and best balance of precision & recall.")
    
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("KNN Accuracy", "92%")
        with col2:
            st.metric("LogReg Accuracy", "87.6%")
        with col3:
            st.metric("Random Forest", "95% ✅")
    
        st.markdown("### 🔹 Confusion Matrix (Random Forest)")
    
        cm = [[14359, 361], [681, 10575]]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Purples", cbar=False,
                    xticklabels=["Predicted: No", "Predicted: Yes"],
                    yticklabels=["Actual: No", "Actual: Yes"])
        ax.set_title("Confusion Matrix – Random Forest")
        st.pyplot(fig)
    
        st.markdown("""
        ### ✅ Why Random Forest?
        - **High Accuracy** (95%) with minimal overfitting  
        - **Precision & Recall** balanced for both classes  
        - Handles both categorical and numerical features well  
        - Feature importance scores help in interpretability
        """)
    
        st.markdown("---")
        st.success("Ready to explore the trends? Head over to the 📊 **EDA** page or maybe make predictions in 🔮 **Predict** page!")

# 📊  EDA PAGE
elif page == "📊 EDA":
    st.title("Exploratory Data Analysis")
    st.markdown("""
    Interactively explore the airline satisfaction dataset below.  
    Use the dropdown to select the plot type and variable(s) you'd like to visualize.
    """)

    # Select Plot Type
    plot_type = st.selectbox("📈 Select Plot Type:", 
                             ["Histogram", "Box Plot", "Scatter Plot"])
    
    # Variable selection logic
    if plot_type == "Histogram":
        col = st.selectbox("🔹 Select Numeric Column", df.select_dtypes(include='number').columns)
        st.subheader(f"Histogram for {col}")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
    
    elif plot_type == "Box Plot":
        col = st.selectbox("🔹 Select Numeric Column", df.select_dtypes(include='number').columns)
        st.subheader(f"Box Plot of {col} by Satisfaction")
        
        if "satisfaction" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x="satisfaction", y=col, data=df, ax=ax)
            st.pyplot(fig)
        else:
            st.error("❌ 'satisfaction' column not found in dataset.")

    elif plot_type == "Scatter Plot":
        col1 = st.selectbox("🔹 X-axis (Numeric)", df.select_dtypes(include='number').columns)
        col2 = st.selectbox("🔸 Y-axis (Numeric)", df.select_dtypes(include='number').columns, index=1)
        st.subheader(f"Scatter Plot: {col1} vs {col2}")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x=col1, y=col2, hue="satisfaction", alpha=0.6, ax=ax)
        st.pyplot(fig)

    # Feature Importance bar graph
    st.markdown("---")
    st.subheader("🌟 Feature Importance (Random Forest Model)")
    
    try:
        # Droping target and non-feature cols if needed
        feature_names = df.drop(columns=["satisfaction"]).columns
        importances = model.feature_importances_
    
        # Creating importance df
        imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by="Importance", ascending=False)
    
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax)
        plt.title("Feature Importance – Random Forest")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Feature importance not available: {e}")

# 🔮 PREDICTION PAGE
elif page == "🔮 Predict":
    st.title("🧠 Airline Satisfaction Predictor")
    st.markdown(
        "Use this tool to predict whether a passenger will be **Satisfied** or **Dissatisfied** based on their flight experience. Just fill in the details below and hit **Predict** ✈️"
    )

    # INPUT
    colA, colB = st.columns(2)

    with colA:
        gender           = st.selectbox("Gender", ["female", "male"])
        customer_type    = st.selectbox("Customer Type", ["loyal customer", "disloyal customer"])
        travel_type      = st.selectbox("Type of Travel", ["business travel", "personal travel"])
        travel_class     = st.selectbox("Class", ["business", "eco plus", "eco"])
        age              = st.slider("Age", 7, 85, 35)
        flight_distance  = st.slider("Flight Distance (mi)", 30, 5000, 1000)

    with colB:
        seat_comfort     = st.slider("Seat Comfort (0–5)", 0, 5, 3)
        inflight_wifi    = st.slider("Inflight Wifi (0–5)", 0, 5, 3)
        online_boarding  = st.slider("Online Boarding (0–5)", 0, 5, 3)
        entertainment    = st.slider("Inflight Entertainment (0–5)", 0, 5, 3)
        cleanliness      = st.slider("Cleanliness (0–5)", 0, 5, 3)
        baggage_handling = st.slider("Baggage Handling (1–5)", 1, 5, 3)
        departure_delay  = st.slider("Departure Delay (min)", 0, 1600, 0)
        arrival_delay    = st.slider("Arrival Delay (min)",   0, 1600, 0)

    # BUILDING INPUT VECTOR MATCHING THE TRAINED MODEL
    # Starting with all-zero frame that has every column the model was trained on
    input_df = pd.DataFrame(
        np.zeros((1, len(feat_order))), columns=feat_order
    )

    #  Helper to set a value only if the column exists
    def set_col(col, val):
        if col in input_df.columns:
            input_df.at[0, col] = val

    # ▸ Numeric Columns
    set_col('Age', age)
    set_col('Flight Distance', flight_distance)
    set_col('Seat comfort', seat_comfort)
    set_col('Inflight wifi service', inflight_wifi)
    set_col('Online boarding', online_boarding)
    set_col('Inflight entertainment', entertainment)
    set_col('Cleanliness', cleanliness)
    set_col('Baggage handling', baggage_handling)
    set_col('Departure Delay in Minutes', departure_delay)
    set_col('Arrival Delay in Minutes',  arrival_delay)

    # If the model includes any engineered columns:
    set_col('Delay_Diff', arrival_delay - departure_delay)
    avg_service = np.mean(
        [seat_comfort, inflight_wifi, online_boarding,
         entertainment, cleanliness, baggage_handling]
    )
    set_col('Avg_Service_Rating', avg_service)

    # One-hot categorical (only set the 1’s)
    set_col('Gender_male',               1 if gender == 'male' else 0)
    set_col('Customer Type_loyal customer',
                1 if customer_type == 'loyal customer' else 0)
    set_col('Type of Travel_personal travel',
                1 if travel_type == 'personal travel' else 0)
    set_col('Class_eco',                 1 if travel_class == 'eco' else 0)
    set_col('Class_eco plus',            1 if travel_class == 'eco plus' else 0)
    set_col('Class_business',            1 if travel_class == 'business' else 0)

    # PREDICTING
    if st.button("🔍 Predict Satisfaction"):
        pred_label = model.predict(input_df)[0]          # 0 / 1
        prob       = model.predict_proba(input_df)[0][pred_label]

        if pred_label == 1:
            st.success(f"✅ Prediction: **Satisfied** (confidence {prob:.2%})")
            st.balloons()
        else:
            st.error(f"⚠️ Prediction: **Neutral / Dissatisfied** "
                     f"(confidence {prob:.2%})")

        # INPUT SUMMARY
        st.markdown("#### Passenger Input Summary")
        summary = {
            "Age": age, "Gender": gender.title(), "Customer": customer_type.title(),
            "Class": travel_class.title(), "Travel Type": travel_type.title(),
            "Flight Distance": flight_distance, "Seat Comfort": seat_comfort,
            "Wifi": inflight_wifi, "Online Boarding": online_boarding,
            "Entertainment": entertainment, "Cleanliness": cleanliness,
            "Baggage Handling": baggage_handling,
            "Delays (Dep/Arr)": f"{departure_delay}/{arrival_delay}"
        }
        st.write(summary)

        # RECOMMENDATIONS
        st.markdown("### ✏️ Recommendations for Airlines:")
        recs = []
        if inflight_wifi < 3:   recs.append("- Improve inflight Wi-Fi service to ensure better connectivity and customer experience.")
        if seat_comfort < 3:    recs.append("- Enhance seat cushioning and legroom to improve comfort.")
        if online_boarding < 3: recs.append("- Simplify and optimize the online boarding system for ease of use.")
        if entertainment < 3:   recs.append("- Offer a wider variety of engaging inflight entertainment options.")
        if cleanliness < 3:     recs.append("- Improve aircraft cleanliness, especially restrooms and seating areas.")
        if baggage_handling < 3:recs.append("- Enhance baggage handling reliability to minimize damage or loss.")
        if departure_delay > 15 or arrival_delay > 15:
            recs.append("- Implement better scheduling and turnaround strategies to reduce delays.")

        if recs:
            for r in recs: st.write(r)
        else:
            st.success("✅ Great! No major service gaps detected for this passenger👍")