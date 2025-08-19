import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================
# Load Dataset
# =========================
df = pd.read_csv("churn_data.csv")

st.title("📊 Customer Churn Prediction App")

st.write("This app predicts whether a customer will churn or not based on their details.")

# =========================
# Data Preprocessing
# =========================
df = df.dropna()

# Convert TotalCharges to numeric (sometimes string)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================
# Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# Train Model
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.sidebar.success(f"✅ Model trained with Accuracy: {acc:.2f}")

# =========================
# Visualizations
# =========================
st.subheader("📈 Data Insights")
fig, ax = plt.subplots()
sns.countplot(x="Churn", data=df, ax=ax)
st.pyplot(fig)

# =========================
# User Input Form
# =========================
st.subheader("🧑‍💻 Try it Yourself: Predict Churn")

def user_input():
    data = {
        "gender": st.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": st.selectbox("Senior Citizen", [0, 1]),
        "Partner": st.selectbox("Partner", ["Yes", "No"]),
        "Dependents": st.selectbox("Dependents", ["Yes", "No"]),
        "tenure": st.slider("Tenure (months)", 0, 72, 24),
        "PhoneService": st.selectbox("Phone Service", ["Yes", "No"]),
        "MultipleLines": st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"]),
        "InternetService": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": st.selectbox("Online Security", ["Yes", "No", "No internet service"]),
        "OnlineBackup": st.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
        "DeviceProtection": st.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
        "TechSupport": st.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
        "StreamingTV": st.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
        "StreamingMovies": st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
        "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": st.selectbox("Paperless Billing", ["Yes", "No"]),
        "PaymentMethod": st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ]),
        "MonthlyCharges": st.number_input("Monthly Charges", 0.0, 200.0, 70.0),
        "TotalCharges": st.number_input("Total Charges", 0.0, 10000.0, 2000.0),
    }
    return pd.DataFrame([data])

user_df = user_input()

# Encode input same way as training data
for col in user_df.select_dtypes(include="object").columns:
    user_df[col] = le.fit_transform(user_df[col])

# 🔑 Ensure input columns match training data
missing_cols = set(X.columns) - set(user_df.columns)
for col in missing_cols:
    user_df[col] = 0  # default value

# Reorder columns exactly like training set
user_df = user_df[X.columns]

# =========================
# Make Prediction
# =========================
if st.button("🔮 Predict Churn"):
    prediction = model.predict(user_df)
    prediction_proba = model.predict_proba(user_df)

    if prediction[0] == 1:
        st.error(f"⚠️ This customer is likely to CHURN! (Probability {prediction_proba[0][1]:.2f})")
    else:
        st.success(f"🎉 This customer is likely to STAY. (Probability {prediction_proba[0][0]:.2f})")
