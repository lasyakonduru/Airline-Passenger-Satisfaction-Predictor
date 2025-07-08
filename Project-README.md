# ✈️ Airline Passenger Satisfaction Classification Project

![Airline](airline.jpg)

Welcome to the Airline Passenger Satisfaction Classification project! In this project, we developed a machine learning-based web application to predict whether airline passengers are satisfied or dissatisfied with their flight experience, based on their flight details and onboard service feedback.

---

## 📌 Objective

The goal of this project is to build a **classification model** that can predict passenger satisfaction (Satisfied / Neutral or Dissatisfied) based on a variety of features, including flight distance, service ratings, delay durations, and demographic data. We aim to deliver an interactive **Streamlit app** that provides both data insights and live prediction capabilities.

---

## 📁 Dataset Overview

- **Source:** [Kaggle - Airline Passenger Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data)
- **Accessed:** February 2025
- **File Used:** `airline_passenger_satisfaction.csv`

### ✅ Features Included
- Demographics: Gender, Age, Customer Type
- Flight info: Type of Travel, Class, Flight Distance
- Service ratings (scale 0–5): WiFi, Food, Seat Comfort, Entertainment, Boarding, Cleanliness, etc.
- Delays: Departure and Arrival Delay in Minutes

### 🎯 Target Variable
- `satisfaction`: Binary classification  
  - `0`: Neutral or Dissatisfied  
  - `1`: Satisfied

---

## 🔍 Part 1: Exploratory Data Analysis (EDA)

### Key Tasks:
- Used `head()`, `info()`, and `describe()` to understand data.
- Identified and imputed **310 missing values** in `Arrival Delay in Minutes` using **median imputation**.
- Visualized distributions of numerical and categorical variables using:
  - **Histograms, Count Plots, Box Plots**
  - **Feature importance bar charts**
- Performed **outlier detection** using boxplots and statistical summaries.

### Notable Insights:
- Business travelers and loyal customers are more likely to be satisfied.
- Key factors affecting satisfaction include:
  - Seat Comfort
  - Online Boarding
  - Inflight WiFi
  - Cleanliness
  - Delay Duration

---

## 🛠️ Part 2: Modeling & Evaluation

### ✨ Feature Engineering
- Encoded categorical variables using Label Encoding and One-Hot Encoding.
- Created new features:
  - `Avg_Service_Rating` (average of all service scores)
  - `Delay_Diff` (Arrival Delay - Departure Delay)

### 🔍 Models Built
| Model                | Train Accuracy | Test Accuracy |
|---------------------|----------------|---------------|
| **K-Nearest Neighbors** (KNN) | 0.94           | 0.92          |
| **Logistic Regression**       | 0.87           | 0.8762        |
| **Random Forest**             | 1.00           | **0.95**       ✅

📌 **Best Model Chosen:** Random Forest (highest accuracy with good generalization and interpretability)

### 🔁 Evaluation Metrics:
- **Confusion Matrix**
- **Classification Report**
- **Feature Importance Plot**

---

## 🌐 Part 3: Streamlit App

### 🔗 Live App:
[🌍 Open Streamlit App](https://your-deployment-link.streamlit.app)

### 🎥 Demo Video:
[▶️ Watch on Loom](https://www.loom.com/share/your-video-link)

### 🧠 App Features:

#### 1️⃣ Home Page
- Welcome message, airline-themed image
- Purpose of the project
- Benefits and potential users (e.g., Airline Quality Teams, Analysts)
- Summary of dataset and Random Forest model performance

#### 2️⃣ EDA Page
- Dropdown menu for visualizing:
  - Histograms
  - Box plots
  - Scatter plots
  - Count plots
  - Feature importance chart
- Celebratory visuals with `st.balloons()` 🎈

#### 3️⃣ Prediction Page
- Interactive sliders for user input:
  - Seat Comfort, Inflight WiFi, Online Boarding, Entertainment, Cleanliness, etc.
- Dynamic prediction result (Satisfied / Not Satisfied)
- Smart recommendations based on inputs (e.g., “Improve WiFi service”, “Reduce delays”)

---

## 🎯 Usage Instructions

1. Clone the repo or download the app files.
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the app:
   ```bash
   streamlit run app.py

## 💡 Recommendations for Airlines
Based on model insights:

- Improve in-flight WiFi and online boarding experience

- Focus on seat comfort and cleanliness

- Reduce both departure and arrival delays

- Encourage loyalty through customer programs

## 👩‍💻 Built With
- Python (pandas, matplotlib, seaborn, scikit-learn)

- Streamlit (for web deployment)

- Jupyter Notebook (for EDA & modeling)

- Loom (for video demo)

## 🙌 Acknowledgments
- Kaggle for providing the dataset

- Coding Temple instructors and peers

- Streamlit & scikit-learn open-source communities

## 📫 Contact
**Lasya Priya Konduru**
[LinkedIn](https://www.linkedin.com/in/lasya-priya-k/) | [GitHub](https://github.com/lasyakonduru) | [Email](konduru.lasya@gmail.com)