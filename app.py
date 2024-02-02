import streamlit as st
import pandas as pd
import numpy as np
import joblib


def main():
    #st.title("Bank Customers Churn")
    html_temp = """
        <div style="background:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">Bank Churn Prediction ML App </h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html = True)

    CustomerId = st.number_input("CustomerId", 0)
    CreditScore = st.number_input("CreditScore", 0)
    Age = st.slider("Age", 10, 100)
    Tenure = st.selectbox("Tenure", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Balance = st.number_input("Balance", 0)
    NumOfProducts = st.selectbox("NumOfProducts", [1, 2, 3, 4])
    HasCrCard = st.selectbox("HasCrCard", [0, 1])
    IsActiveMember = st.selectbox("IsActiveMember", [0, 1])
    EstimatedSalary = st.number_input("EstimatedSalary", 0)
    Geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    Gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("Predict"):
        models   = ["hist", "cat", "gbm", "lgbm", "xgb"]
        features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary", "Geography", "Gender"]
        row = np.array([CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography, Gender])
        X = pd.DataFrame([row], columns = features)
        predictions = None
        threshold = 0.5

        mapping_Geography = {"France": 0, "Spain": 1, "Germany": 2}
        mapping_Gender = {"Male": 0, "Female": 1}
        X.loc[:, "Geography"] = X.Geography.map(mapping_Geography)
        X.loc[:, "Gender"] = X.Gender.map(mapping_Gender)
        X = X[features].values

        for model in models:
            for fold in range(5):
                clf = joblib.load(f"./models/{model}_{fold}.pkl")
                preds = clf.predict_proba(X)[:, 1]
                if fold == 0:
                    predictions = preds
                else:
                    predictions += preds
            predictions = predictions / 5

            if model == 'hist':
                ens_preds = predictions
            else:
                ens_preds += predictions

        ens_preds = ens_preds / 5
        output = ens_preds.astype(float)

        if output > threshold:
            st.success(f"Predicted Probability : {output}. \nThis customer is likely to churn :thumbsdown:")
        else:
            st.success(f"Predicted Probability : {output}. \nThis customer isn't likely to churn :thumbsup:")

if __name__=='__main__': 
    main()