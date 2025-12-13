import pandas as pd
from sklearn.naive_bayes import GaussianNB
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

model = GaussianNB()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

print(f"\nThe test accuracy score is : {accuracy_score(y_test.values, predictions)}")

with open("C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/classification/trained_models/gaussian_nb_classifier.pkl", "wb") as file:
    pickle.dump(model, file)
