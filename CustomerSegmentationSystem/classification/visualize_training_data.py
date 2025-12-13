import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import math

df = pd.read_csv(
    filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/PREDICTORS_TRAIN.csv",
    header=0,
    sep=",")

y = pd.read_csv(
    filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/RESPONSE_TRAIN.csv",
    header=0,
    sep=",")

number_of_columns = df.columns.size
lines = math.ceil(math.sqrt(number_of_columns))

plt.figure(figsize=(10, 8))
for index, col in zip(range(1, number_of_columns + 1), df.columns):
    plt.subplot(lines, lines, index)
    plt.title(col)
    sb.histplot(x=df[col], kde=True, bins=20)
    plt.grid(True)
plt.tight_layout()
plt.show()

# we see feature values are already normalized, so we don't do further scalling. We keep data as it is
# we need to make sure there's no CLASS IMBALANCE PROBLEM

plt.figure(figsize=(10, 10))
sb.countplot(x=y.CLUSTER_ID, hue=y.CLUSTER_ID, palette="muted")
plt.tight_layout()
plt.grid(True)
plt.show()

# we see cluster id = 2 has relatively lower number of samples, although not too small
