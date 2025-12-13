import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, RandomizedSearchCV
import pickle
import math
import warnings

warnings.filterwarnings(action="ignore")

x_train = pd.read_csv(
    filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/PREDICTORS_TRAIN.csv",
    header=0,
    sep=",")

x_test = pd.read_csv(
    filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/PREDICTORS_TEST.csv",
    header=0,
    sep=",")

y_train = pd.read_csv(
    filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/RESPONSE_TRAIN.csv",
    header=0,
    sep=",")

y_test = pd.read_csv(
    filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/RESPONSE_TEST.csv",
    header=0,
    sep=",")

y_train = y_train["CLUSTER_ID"]
y_test = y_test["CLUSTER_ID"]

hyperparameters = {
    "n_estimators": np.arange(start=25, stop=151, step=25),
    "learning_rate": np.logspace(start=-1.5, stop=0.9, num=25),
    "max_depth": np.arange(start=3, stop=16, step=1),
    "reg_lambda": np.logspace(start=-1, stop=1, num=25),
    "gamma": np.logspace(start=-1.5, stop=1.5, num=25)
}

model = XGBClassifier(n_jobs=-1)

best_selected_model = None
best_selected_params = None
best_selected_score = -math.inf

for i in range(100):
    grid_search_cv = RandomizedSearchCV(estimator=model,
                                        scoring="accuracy",
                                        cv=KFold(n_splits=5, shuffle=True),
                                        n_jobs=-1,
                                        verbose=True,
                                        param_distributions=hyperparameters)

    grid_search_cv.fit(x_train, y_train)

    best_model = grid_search_cv.best_estimator_
    best_params = grid_search_cv.best_params_
    best_score = grid_search_cv.best_score_

    if best_score > best_selected_score:
        best_selected_score = best_score
        best_selected_params = best_params
        best_selected_model = best_model

print(f"\nThe best parameters are {best_selected_params} with best cross-validation accuracy score : {best_selected_score}")

predictions = best_selected_model.predict(x_test)
score = accuracy_score(y_test, predictions)

print(f"\nThe testing accuracy score is : {score}")

with open(
        "C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/classification/trained_models/xgboost_classifier.pkl",
        "wb") as file:
    pickle.dump(best_selected_model, file)
