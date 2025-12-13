import warnings
import pickle
import pandas as pd

warnings.filterwarnings(action="ignore")


class FinalClassifierModel:

    def __init__(self):
        self.__models = []
        self.__base_path = "C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/classification/trained_models/"

        logistic_regressor = None
        with open(self.__base_path + "logistic_regression.pkl", "rb") as file:
            logistic_regressor = pickle.load(file)
        self.__models.append(logistic_regressor)

        support_vector_classifier = None
        with open(self.__base_path + "support_vector_classifier.pkl", "rb") as file:
            support_vector_classifier = pickle.load(file)
        self.__models.append(support_vector_classifier)

        gaussian_nb_classifier = None
        with open(self.__base_path + "gaussian_nb_classifier.pkl", "rb") as file:
            gaussian_nb_classifier = pickle.load(file)
        self.__models.append(gaussian_nb_classifier)

        random_forest_classifier = None
        with open(self.__base_path + "random_forest_classifier.pkl", "rb") as file:
            random_forest_classifier = pickle.load(file)
        self.__models.append(random_forest_classifier)

        gradient_boost_classifier = None
        with open(self.__base_path + "gradient_boost_classifier.pkl", "rb") as file:
            gradient_boost_classifier = pickle.load(file)
        self.__models.append(gradient_boost_classifier)

        xgboost_classifier = None
        with open(self.__base_path + "xgboost_classifier.pkl", "rb") as file:
            xgboost_classifier = pickle.load(file)
        self.__models.append(xgboost_classifier)

        adaboost_classifier = None
        with open(self.__base_path + "adaboost_classifier.pkl", "rb") as file:
            adaboost_classifier = pickle.load(file)
        self.__models.append(adaboost_classifier)

    def predict(self, x):

        predictions = []

        for model in self.__models:
            predictions.append(model.predict(x))

        r_size = len(predictions)
        c_size = len(predictions[0])

        final_predictions = []
        for c in range(c_size):

            vote_to_take = -1
            max_count = -1

            for r in range(r_size):

                votes = {}

                val = predictions[r][c]
                if val in votes.keys():
                    votes[val] += 1
                else:
                    votes[val] = 1

                for cluster_id, count in votes.items():
                    if count > max_count:
                        max_count = count
                        vote_to_take = cluster_id

            final_predictions.append(vote_to_take)

        return final_predictions


x_test = pd.read_csv(
    filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/PREDICTORS_TEST.csv",
    header=0,
    sep=",")

y_test = pd.read_csv(
    filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/RESPONSE_TEST.csv",
    header=0,
    sep=",")

y_test = y_test["CLUSTER_ID"]

model = FinalClassifierModel()
with open("C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/classification/trained_models/final_classifier.pkl", "wb") as file:
    pickle.dump(model, file)
