import pandas as pd

data=pd.read_csv("dataset.csv")

X = data.drop(['customerID', 'Churn'], axis=1) 
y = data['Churn'].map({'Yes':1, 'No':0})       

categorical_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService',
                    'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                    'StreamingMovies','Contract','PaperlessBilling','PaymentMethod']

numeric_cols = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']

X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
X[numeric_cols] = X[numeric_cols].fillna(0)  
X_encoded = pd.get_dummies(X, columns=categorical_cols)
print(X_encoded.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=1
)

print(X_train.shape, X_test.shape) 

#standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(
    hidden_layer_sizes=(45, 45, 45),  
    max_iter=500,                     
    random_state=1
)

mlp_model.fit(X_train_scaled, y_train)

#evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

y_pred = mlp_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


import joblib

joblib.dump(mlp_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X_encoded.columns), "columns.pkl")

print("Model doneeee")


def predict_new_customer(input_dict):
    import pandas as pd
    import joblib

    # load saved artifacts
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")

    df = pd.DataFrame([input_dict])

    # one-hot encode to match training
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    # scale
    df_scaled = scaler.transform(df)

    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    return pred, prob

print("\nENTER CUSTOMER DETAILS ")

def inp(prompt):
    return input(prompt + ": ")

customer = {
    "gender": inp("Gender (Male/Female)"),
    "SeniorCitizen": int(inp("Senior Citizen? (0 = No, 1 = Yes)")),
    "Partner": inp("Partner (Yes/No)"),
    "Dependents": inp("Dependents (Yes/No)"),
    "tenure": int(inp("Tenure (months)")),
    "PhoneService": inp("Phone Service (Yes/No)"),
    "MultipleLines": inp("Multiple Lines (Yes/No/No phone service)"),
    "InternetService": inp("Internet Service (DSL/Fiber optic/No)"),
    "OnlineSecurity": inp("Online Security (Yes/No/No internet service)"),
    "OnlineBackup": inp("Online Backup (Yes/No/No internet service)"),
    "DeviceProtection": inp("Device Protection (Yes/No/No internet service)"),
    "TechSupport": inp("Tech Support (Yes/No/No internet service)"),
    "StreamingTV": inp("Streaming TV (Yes/No/No internet service)"),
    "StreamingMovies": inp("Streaming Movies (Yes/No/No internet service)"),
    "Contract": inp("Contract (Month-to-month/One year/Two year)"),
    "PaperlessBilling": inp("Paperless Billing (Yes/No)"),
    "PaymentMethod": inp("Payment Method (Electronic check/Mailed check/Bank transfer/Credit card)"),
    "MonthlyCharges": float(inp("Monthly Charges (e.g., 75.5)")),
    "TotalCharges": float(inp("Total Charges (e.g., 300.0)"))
}

pred, prob = predict_new_customer(customer)

print("\n=== RESULT ===")
print("Churn:", " YES (customer likely to leave)" if pred == 1 else "🟢 NO (customer will stay)")
print("Probability:", round(prob, 4))