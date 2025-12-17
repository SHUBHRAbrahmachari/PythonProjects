import os
import pickle
import pandas as pd
from pymongo import MongoClient
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings(action="ignore")

client = MongoClient(os.getenv("MONGODB_URL"))
db = client["PhishingClassifierProject"]

training_records = db["processed_train_data"]
df_training = pd.DataFrame(data=training_records.find({}))
df_training.drop(columns=["_id"], inplace=True)

testing_records = db["processed_test_data"]
df_testing = pd.DataFrame(data=testing_records.find({}))
df_testing.drop(columns=["_id"], inplace=True)

print(df_training, '\n\n', df_testing)

x_train = df_training.drop(columns=["Result"], inplace=False)
y_train = df_training["Result"]

x_test = df_testing.drop(columns=["Result"], inplace=False)
y_test = df_testing["Result"]

model = CatBoostClassifier(verbose=False,
                           random_state=42)

model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(f"\nThe testing accuracy score is : {accuracy_score(y_test.values, predictions)}\n")
print(f"\nThe confusion matrix is like that : \n{confusion_matrix(y_test.values, predictions)}\n")
with open("C:/Users/Shubhra/PycharmProjects/PhishingClassifierSystem/trained_models/catboost_classifier.pkl",
          mode="wb") as file:
    pickle.dump(model, file)

print(f"\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tMODEL SAVED!\n\n")
