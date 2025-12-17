import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder="./templates")

model = None

try:
    with open("C:/Users/Shubhra/PycharmProjects/PhishingClassifierSystem/trained_models/best_model.pkl",
              "rb") as file:
        model = pickle.load(file)
except Exception as e:
    print(e.args)
    exit(1)
else:
    print("Model loaded successfully!")


@app.route("/api/home/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/api/predict/", methods=["POST"])
def predict():
    data = {
        "having_IP_Address": [int(request.form.get("having_IP_Address"))],
        "URL_Length": [int(request.form.get("URL_Length"))],
        "Shortining_Service": [int(request.form.get("Shortining_Service"))],
        "having_At_Symbol": [int(request.form.get("having_At_Symbol"))],
        "double_slash_redirecting": [int(request.form.get("double_slash_redirecting"))],
        "Prefix_Suffix": [int(request.form.get("Prefix_Suffix"))],
        "having_Sub_Domain": [int(request.form.get("having_Sub_Domain"))],
        "SSLfinal_State": [int(request.form.get("SSLfinal_State"))],
        "Domain_registeration_length": [int(request.form.get("Domain_registeration_length"))],
        "Favicon": [int(request.form.get("Favicon"))],
        "port": [int(request.form.get("port"))],
        "HTTPS_token": [int(request.form.get("HTTPS_token"))],
        "Request_URL": [int(request.form.get("Request_URL"))],
        "URL_of_Anchor": [int(request.form.get("URL_of_Anchor"))],
        "Links_in_tags": [int(request.form.get("Links_in_tags"))],
        "SFH": [int(request.form.get("SFH"))],
        "Submitting_to_email": [int(request.form.get("Submitting_to_email"))],
        "Abnormal_URL": [int(request.form.get("Abnormal_URL"))],
        "Redirect": [int(request.form.get("Redirect"))],
        "on_mouseover": [int(request.form.get("on_mouseover"))],
        "RightClick": [int(request.form.get("RightClick"))],
        "popUpWidnow": [int(request.form.get("popUpWidnow"))],
        "Iframe": [int(request.form.get("Iframe"))],
        "age_of_domain": [int(request.form.get("age_of_domain"))],
        "DNSRecord": [int(request.form.get("DNSRecord"))],
        "web_traffic": [int(request.form.get("web_traffic"))],
        "Page_Rank": [int(request.form.get("Page_Rank"))],
        "Google_Index": [int(request.form.get("Google_Index"))],
        "Links_pointing_to_page": [int(request.form.get("Links_pointing_to_page"))],
        "Statistical_report": [int(request.form.get("Statistical_report"))]
    }

    print(data)

    df = pd.DataFrame(data=data)
    print(df)
    prediction = model.predict(df)

    return render_template("prediction.html",
                           prediction="THIS WEBSITE IS NOT A PHISHING WEBSITE!") if prediction == 0 else render_template(
        "prediction.html", prediction="THIS WEBSITE IS A PHISHING WEBSITE!")


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True, load_dotenv=True)
