import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction System")

# Load Dataset
df = pd.read_csv("heart.csv")

# Data Cleaning
df = df.drop_duplicates()
df = df.fillna(df.mean(numeric_only=True))

# Feature Selection
corr = df.corr()['target'].abs()
features = corr[corr > 0.1].index.drop('target')

# Split Data
X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

st.success(f"Model Accuracy: {accuracy*100:.2f}%")

st.subheader("Enter Patient Details")

user_input = {}

for feature in features:
    user_input[feature] = st.number_input(feature, value=float(df[feature].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

st.subheader("Batch Prediction")

uploaded_file = st.file_uploader("Upload test_data.csv", type="csv")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    if 'target' in test_df.columns:
        test_df = test_df.drop(columns=['target'])

    test_df = test_df[features]

    predictions = model.predict(test_df)

    test_df["Prediction"] = predictions

    st.write(test_df)

    csv = test_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "Download Results",
        csv,
        "prediction_results.csv",
        "text/csv"
    )
