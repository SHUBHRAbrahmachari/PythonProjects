import math
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

df1 = pd.read_csv(filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/marketing_campaign_numerical_processed.csv",
                  header=0,
                  sep=",")

df2 = pd.read_csv(filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/marketing_campaign_categorical_processed.csv",
                  header=0,
                  sep=",")

df = pd.concat([df1, df2], axis=1)

number_of_columns = df.columns.size
size = math.ceil(math.sqrt(number_of_columns))

plt.figure(figsize=(15, 9))
for index, col in zip(range(1, number_of_columns+1), df.columns):
    plt.subplot(size, size, index)
    sb.histplot(x=df[col], kde=True, bins=20)
    plt.title(col)
    plt.tight_layout()
    plt.grid(True)
plt.show()


plt.figure(figsize=(15, 9))
for index, col in zip(range(1, number_of_columns+1), df.columns):
    plt.subplot(size, size, index)
    sb.boxplot(x=df[col])
    plt.title(col)
    plt.tight_layout()
    plt.grid(True)
plt.show()

df.to_csv(path_or_buf="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/total_marketing_campaign_processed.csv",
          header=True,
          index=False,
          sep=",")
