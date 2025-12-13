import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/K_MEANS_RESULT.csv",
    header=0,
    sep=",")

x = df.drop(columns=["CLUSTER_ID"], inplace=False)
y = df["CLUSTER_ID"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

x_train.to_csv(path_or_buf="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/PREDICTORS_TRAIN.csv",
               header=True,
               index=False,
               sep=",")

x_test.to_csv(path_or_buf="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/PREDICTORS_TEST.csv",
              header=True,
              index=False,
              sep=",")

y_train.to_csv(path_or_buf="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/RESPONSE_TRAIN.csv",
               header=True,
               index=False,
               sep=",")

y_test.to_csv(path_or_buf="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/RESPONSE_TEST.csv",
              header=True,
              index=False,
              sep=",")
