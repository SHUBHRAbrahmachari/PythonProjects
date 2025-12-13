import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
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

model = LogisticRegression(n_jobs=-1,
                           fit_intercept=True,
                           random_state=42)

hyperparameters = {
    "penalty": np.array(["l1", "l2", "elasticnet"]),
    "C": np.array([1/x for x in np.logspace(start=-2, stop=2, num=100)]),
    "l1_ratio": np.logspace(start=-2, stop=-1, num=100)
}

grid_search_cv = GridSearchCV(estimator=model,
                              cv=KFold(n_splits=5, shuffle=True, random_state=42),
                              scoring="accuracy",
                              verbose=True,
                              param_grid=hyperparameters)

grid_search_cv.fit(x_train, y_train)

best_model = grid_search_cv.best_estimator_
best_score = grid_search_cv.best_score_
best_params = grid_search_cv.best_params_

print(f"\nThe best parameters are : {best_params}\n with best cross validation score : {best_score}\n")

predictions = best_model.predict(x_test)

print(f"\nThe accuracy score is : {accuracy_score(y_test, predictions)}\n")
print(f"\nThe confusion matrix is : \n{confusion_matrix(y_test, predictions)}\n")

with open("C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/classification/trained_models/logistic_regression.pkl", "wb") as file:
    pickle.dump(best_model, file)
