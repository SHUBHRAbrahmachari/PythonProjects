import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
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

hyperparameters = {
    "n_estimators": np.arange(start=25, stop=251, step=15),
    "learning_rate": np.logspace(start=-2, stop=0.9, num=100),
    "max_depth": np.arange(start=3, stop=16, step=1),
    "min_samples_leaf": np.arange(start=2, stop=6, step=1),
    "ccp_alpha": np.logspace(start=-1.5, stop=0.99, num=50)
}

model = GradientBoostingClassifier(random_state=42)

grid_search_cv = RandomizedSearchCV(estimator=model,
                                    scoring="accuracy",
                                    cv=KFold(n_splits=5, shuffle=True),
                                    n_jobs=-1,
                                    verbose=True,
                                    param_distributions=hyperparameters,
                                    n_iter=100)

grid_search_cv.fit(x_train, y_train)

best_model = grid_search_cv.best_estimator_
best_params = grid_search_cv.best_params_
best_score = grid_search_cv.best_score_

print(
    f"\nThe best parameters are {best_params} with best cross-validation accuracy score : {best_score}")

predictions = best_model.predict(x_test)
score = accuracy_score(y_test, predictions)

print(f"\nThe testing accuracy score is : {score}")

with open(
        "C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/classification/trained_models/gradient_boost_classifier.pkl",
        "wb") as file:
    pickle.dump(best_model, file)
