import os
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.best_model import BestModel
import warnings
import pickle

warnings.filterwarnings(action="ignore")

base_path = "C:/Users/Shubhra/PycharmProjects/PhishingClassifierSystem/trained_models/"

model_names = os.listdir(path=base_path)

print(model_names)

client = MongoClient(os.getenv("MONGODB_URL"))
db = client.PhishingClassifierProject

processed_test_data = db.processed_test_data.find({})
df = pd.DataFrame(data=processed_test_data)
df.drop(columns=["_id"], axis=1, inplace=True)
x = df.drop(columns=["Result"], inplace=False)
y = df["Result"]

client.close()

rows = []
for model_name in model_names:
    model = None
    with open(base_path + model_name, mode="rb") as file:
        model = pickle.load(file)

    predictions = model.predict(x)

    row = {}

    # appending MODEL_NAME
    size = len(model_name)
    row["MODEL_NAME"] = model_name[0: size - 4: 1]

    # appending ACCURACY_SCORE
    row["ACCURACY_SCORE"] = accuracy_score(y.values, predictions)

    # appending PRCISION_SCORE_0
    row["PRECISION_SCORE_0"] = precision_score(y.values, predictions, pos_label=0)

    # appending PRCISION_SCORE_1
    row["PRECISION_SCORE_1"] = precision_score(y.values, predictions, pos_label=1)

    # appending RECALL_SCORE_0
    row["RECALL_SCORE_0"] = recall_score(y.values, predictions, pos_label=0)

    # appending RECALL_SCORE_1
    row["RECALL_SCORE_1"] = recall_score(y.values, predictions, pos_label=1)

    # appending F1_SCORE_0
    row["F1_SCORE_0"] = f1_score(y.values, predictions, pos_label=0)

    # appending F1_SCORE_1
    row["F1_SCORE_1"] = f1_score(y.values, predictions, pos_label=1)

    rows.append(row)

# got the dataframe
performances = pd.DataFrame(data=rows)
# here detecting spam mails are more important.... here we would go with f1_score with x_pos_label = 1
performances.sort_values(["F1_SCORE_1", "RECALL_SCORE_1", "PRECISION_SCORE_1"], ascending=[False, False, False],
                         inplace=True, ignore_index=True)
top_performance = performances.iloc[0: 3]  # selecting top 3 models


best_model = BestModel(top_performance)

with open("C:/Users/Shubhra/PycharmProjects/PhishingClassifierSystem/trained_models/best_model.pkl", mode="wb") as file:
    pickle.dump(best_model, file)
