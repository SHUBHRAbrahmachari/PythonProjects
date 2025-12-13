import pandas
import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime
import pickle
import warnings

warnings.filterwarnings(action="ignore")

app = Flask(__name__, template_folder="./templates")


@app.route(rule="/api/home", methods=["GET"])
def home():
    return render_template("home.html")


@app.route(rule="/api/predict", methods=["POST"])
def predict():
    data = {"Year_Birth": [int(request.form.get("year-birth"))],
            "Education": [request.form.get("education")],
            "Marital_Status": [request.form.get("Marital_Status")],
            "Income": [float(request.form.get("Income"))],
            "Kidhome": [int(request.form.get("Kidhome"))],
            "Teenhome": [int(request.form.get("Teenhome"))],
            "Dt_Customer": [request.form.get("Dt_Customer")],
            "Recency": [int(request.form.get("Recency"))],
            "MntWines": [int(request.form.get("MntWines"))],
            "MntFruits": [int(request.form.get("MntFruits"))],
            "MntMeatProducts": [int(request.form.get("MntMeatProducts"))],
            "MntFishProducts": [int(request.form.get("MntFishProducts"))],
            "MntSweetProducts": [int(request.form.get("MntSweetProducts"))],
            "MntGoldProds": [int(request.form.get("MntGoldProds"))],
            "NumDealsPurchases": [int(request.form.get("NumDealsPurchases"))],
            "NumCatalogPurchases": [int(request.form.get("NumCatalogPurchases"))],
            "NumStorePurchases": [int(request.form.get("NumStorePurchases"))],
            "NumWebVisitsMonth": [int(request.form.get("NumWebVisitsMonth"))],
            "NumWebPurchases": [int(request.form.get("NumWebPurchases"))],
            "AcceptedCmp1": [int(request.form.get("AcceptedCmp1"))],
            "AcceptedCmp2": [int(request.form.get("AcceptedCmp2"))],
            "AcceptedCmp3": [int(request.form.get("AcceptedCmp3"))],
            "AcceptedCmp4": [int(request.form.get("AcceptedCmp4"))],
            "AcceptedCmp5": [int(request.form.get("AcceptedCmp5"))],
            "Complain": [int(request.form.get("Complain"))],
            "Response": [int(request.form.get("Response"))]}
    model = None
    with open(
            "C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/classification/trained_models/final_classifier.pkl",
            "rb") as file:
        model = pickle.load(file)

    df = pandas.DataFrame(data=data)

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%Y-%m-%d")
    df["Customer_Duration"] = datetime.now() - df["Dt_Customer"]
    df["Customer_Duration_Days"] = df["Customer_Duration"].dt.days

    df.drop(columns=["Dt_Customer", "Customer_Duration"], inplace=True)

    numerical_transformer = None
    with open("C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/eda/transformers/numerical_transformer.pkl",
              "rb") as file:
        numerical_transformer = pickle.load(file)

    categorical_transformer = None
    with open(
            "C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/eda/transformers/categorical_transformer.pkl",
            "rb") as file:
        categorical_transformer = pickle.load(file)

    df_numerical = df[
        ["Year_Birth", "Income", "Kidhome", "Teenhome", "Recency", "MntWines", "MntFruits", "MntMeatProducts",
         "MntFishProducts", "MntSweetProducts", "MntGoldProds", "NumDealsPurchases", "NumWebPurchases",
         "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "AcceptedCmp1", "AcceptedCmp2",
         "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Complain", "Response"]]
    df_categorical = df[["Education", "Marital_Status", "Customer_Duration_Days"]]

    df_numerical_transformed = pd.DataFrame(numerical_transformer.transform(df_numerical),
                                            columns=numerical_transformer.get_feature_names_out())
    df_categorical_transformed = pd.DataFrame(categorical_transformer.transform(df_categorical),
                                              columns=categorical_transformer.get_feature_names_out())

    df_processed = pd.concat([df_numerical_transformed, df_categorical_transformed], axis=1)

    prediction = model.predict(df_processed)[0]

    return render_template("prediction.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
