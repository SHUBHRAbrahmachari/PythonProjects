import os
import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings(action="ignore")

client = MongoClient(os.getenv("MONGODB_URL"))
db = client["PhishingClassifierProject"]

training_records = db.processed_train_data
df_training = pd.DataFrame(data=training_records.find({}))
df_training.drop(columns=["_id"], inplace=True)

testing_records = db.processed_test_data
df_testing = pd.DataFrame(data=testing_records.find({}))
df_testing.drop(columns=["_id"], inplace=True)

print(df_training, '\n\n', df_testing)

x_train = df_training.drop(columns=["Result"], inplace=False)
y_train = df_training["Result"]

x_test = df_testing.drop(columns=["Result"], inplace=False)
y_test = df_testing["Result"]

model = LogisticRegression(n_jobs=-1,
                           random_state=42,
                           penalty="elasticnet",
                           fit_intercept=True,
                           solver="saga")

hyperparameters = {
    "C": np.array([1/x for x in np.logspace(start=-1.5, stop=1.5, num=100, base=10)]),
    "l1_ratio": np.logspace(start=-2, stop=-0.1, num=100, base=100)
}

cross_validator = RandomizedSearchCV(estimator=model,
                                     param_distributions=hyperparameters,
                                     cv=KFold(n_splits=5,
                                              shuffle=True,
                                              random_state=42),
                                     scoring="accuracy",
                                     n_jobs=-1,
                                     verbose=True,
                                     n_iter=1000)

cross_validator.fit(x_train, y_train)

best_estimator = cross_validator.best_estimator_
best_params = cross_validator.best_params_
best_score = cross_validator.best_score_

print(f"\nThe best parameters are : {best_params} with best cross-validation score : {best_score}\n")

predictions = best_estimator.predict(x_test)

print(f"\nThe testing accuracy score is : {accuracy_score(y_test.values, predictions)}\n")
print(f"\nThe confusion matrix is like that : \n{confusion_matrix(y_test.values, predictions)}\n")

with open("C:/Users/Shubhra/PycharmProjects/PhishingClassifierSystem/trained_models/logistic_regressor.pkl", mode="wb") as file:
    pickle.dump(best_estimator, file)

print(f"\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tMODEL SAVED!\n")
