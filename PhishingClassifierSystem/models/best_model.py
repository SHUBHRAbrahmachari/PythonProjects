import pickle
from dataclasses import dataclass
import pandas as pd


@dataclass(repr=True, init=False, eq=True, order=True)
class BestModel:

    def __init__(self, top_performance):
        self.__models = top_performance.MODEL_NAME.tolist()
        print(f"\nThe top models are : {self.__models}\n")
        self.__transformer = None
        with open("C:/Users/Shubhra/PycharmProjects/PhishingClassifierSystem/transformers/column_transformer.pkl",
                  "rb") as file:
            self.__transformer = pickle.load(file)

        extension = ".pkl"
        model_base_path = "C:/Users/Shubhra/PycharmProjects/PhishingClassifierSystem/trained_models/"

        self.__model1 = None
        with open(model_base_path + self.__models[0] + extension, mode="rb") as file:
            self.__model1 = pickle.load(file)

        self.__model2 = None
        with open(model_base_path + self.__models[1] + extension, mode="rb") as file:
            self.__model2 = pickle.load(file)

        self.__model3 = None
        with open(model_base_path + self.__models[2] + extension, mode="rb") as file:
            self.__model3 = pickle.load(file)

    def predict(self, X):
        transformed_x = pd.DataFrame(data=self.__transformer.transform(X),
                                     columns=self.__transformer.get_feature_names_out())
        print(transformed_x)

        prediction1 = self.__model1.predict(transformed_x)[0]
        prediction2 = self.__model2.predict(transformed_x)[0]
        prediction3 = self.__model3.predict(transformed_x)[0]

        votes = {1: 0, 0: 0}

        if prediction1 == 1:
            votes[1] += 1
        else:
            votes[0] += 1

        if prediction2 == 1:
            votes[1] += 1
        else:
            votes[0] += 1

        if prediction3 == 1:
            votes[1] += 1
        else:
            votes[0] += 1

        class_to_predict = -1
        vote_count = -1

        for k, v in votes.items():
            if v > vote_count:
                vote_count = v
                class_to_predict = k

        return class_to_predict
