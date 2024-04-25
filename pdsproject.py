import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

@st.cache_data
def load_data():
    df = pd.read_csv('lending_data.csv')
    df.fillna(df.median(), inplace=True)
    return df

def train_models(df):
    X = df[['borrower_income', 'total_debt', 'num_of_accounts', 'loan_size']]
    y = df['loan_status']
    st.write("Class blance:", y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for model in models.values():
        model.fit(X_train_scaled, y_train)
    return models, scaler, X_test_scaled, y_test

# Compute class weights
df = load_data()
class_weights = compute_class_weight('balanced', classes=np.unique(df['loan_status']), y=df['loan_status'])
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Include class_weight in RandomForestClassifier
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights_dict),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}


models, scaler, X_test_scaled, y_test = train_models(df)

with st.form("input_form"):
    borrower_income = st.number_input("Enter your income", min_value=1000, max_value=100000, value=50000)
    total_debt = st.number_input("Enter your debt levels", min_value=0, max_value=100000, value=10000)
    num_of_accounts = st.number_input("Enter the number of accounts", min_value=1, max_value=30, value=5)
    loan_size = st.number_input("Enter your loan size", min_value=1000, max_value=500000, value=25000)
    submitted = st.form_submit_button("Predict Loan Status")

if submitted:
    features = np.array([[borrower_income, total_debt, num_of_accounts, loan_size]])
    features_scaled = scaler.transform(features)
    st.subheader('Predictions and Diagnostics')
    for name, model in models.items():
        prediction = model.predict(features_scaled)
        proba = model.predict_proba(features_scaled)
        st.write(f'{name} Prediction:', 'Success' if prediction[0] == 1 else 'Fail', f'with probability {proba[0][1]:.2f} of Success')

        # Display model evaluation on test set
        y_pred = model.predict(X_test_scaled)
        st.write(f"{name} - Test set accuracy: ", model.score(X_test_scaled, y_test))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))
