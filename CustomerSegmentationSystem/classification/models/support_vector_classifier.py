import math
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
import pickle

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

linear_hyperparameters = {
    "C": np.logspace(start=-2, stop=2, num=200)
}

poly_hyperparameters = {
    "C": np.logspace(start=-2, stop=2, num=150),
    "coef0": np.logspace(start=-1.5, stop=1.8, num=100),
    "degree": np.arange(start=1, stop=5, step=1),
}

rbf_hyperparameters = {
    "C": np.logspace(start=-2, stop=2, num=150),
    "gamma": np.logspace(start=-3, stop=1, num=100)
}

linear_grid_search_cv = GridSearchCV(estimator=SVC(random_state=42,
                                                   probability=False),
                                     param_grid=linear_hyperparameters,
                                     scoring="accuracy",
                                     cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                     n_jobs=-1,
                                     verbose=True)

poly_grid_search_cv = GridSearchCV(estimator=SVC(random_state=42,
                                                 probability=False),
                                   param_grid=poly_hyperparameters,
                                   scoring="accuracy",
                                   cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                   n_jobs=-1,
                                   verbose=True)

rbf_grid_search_cv = GridSearchCV(estimator=SVC(random_state=42,
                                                probability=False),
                                  param_grid=rbf_hyperparameters,
                                  scoring="accuracy",
                                  cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                  n_jobs=-1,
                                  verbose=True)

linear_grid_search_cv.fit(x_train, y_train)
poly_grid_search_cv.fit(x_train, y_train)
rbf_grid_search_cv.fit(x_train, y_train)

best_linear_kernel_model = linear_grid_search_cv.best_estimator_
best_poly_kernel_model = poly_grid_search_cv.best_estimator_
best_rbf_kernel_model = rbf_grid_search_cv.best_estimator_

best_score = -math.inf
best_estimator = None

models = [best_linear_kernel_model, best_poly_kernel_model, best_rbf_kernel_model]

for model in models:

    predictions = model.predict(x_test)
    score = accuracy_score(y_test, predictions)
    if score > best_score:
        best_estimator = model
        best_score = score

with open("C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/classification/trained_models/support_vector_classifier.pkl", "wb") as file:
    pickle.dump(best_estimator, file)
