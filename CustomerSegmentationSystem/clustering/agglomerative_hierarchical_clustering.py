import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings(action="ignore")


def get_wcss_score(dataset, labelset):
    x = dataset.copy()
    columns = x.columns
    x["CLUSTER_ID"] = labelset
    wcss_score = 0

    clusters = x.CLUSTER_ID.unique()

    for cluster in clusters:
        x_selected = x[x.CLUSTER_ID == cluster]
        size = x_selected.shape[0]

        centriod_points = {}

        for col in columns:
            centriod_points[col] = np.sum(x_selected[col].values) / size

        for index in range(x_selected.shape[0]):
            squared_sum = 0
            for col in columns:
                squared_sum += (x_selected[col].iloc[index] - centriod_points[col]) ** 2
            wcss_score += squared_sum

    return wcss_score


df = pd.read_csv(
    filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/total_marketing_campaign_processed.csv",
    header=0,
    sep=",")

ks = np.arange(2, 11, 1)
silhouette_scores = []
wcss_scores = []

for k in ks:
    model = AgglomerativeClustering(n_clusters=k)
    model.fit(df)
    score = silhouette_score(df, model.labels_)
    silhouette_scores.append(score)
    wcss_scores.append(get_wcss_score(df, model.labels_))

plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
sb.lineplot(x=ks, y=silhouette_scores, color="red")
plt.title("Silhouette Score")
plt.grid(True)

plt.subplot(1, 2, 2)
sb.lineplot(x=ks, y=wcss_scores, color="blue")
plt.title("WCSS Score")
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nWe agree with k = 5 i.e. 5 clusters")

final_model = AgglomerativeClustering(n_clusters=5)
labels = final_model.fit_predict(df)

print(f"\nThe silhouette score is : {silhouette_score(df, labels)}")

df["CLUSTER_ID"] = labels

df.to_csv(path_or_buf="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/AHC_RESULT.csv",
          index=False,
          header=True,
          sep=",")
