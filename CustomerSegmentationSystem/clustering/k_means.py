import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings(action="ignore")

df = pd.read_csv(filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/total_marketing_campaign_processed.csv",
                 header=0,
                 sep=",")

ks = np.arange(2, 11, 1)
silhouette_scores = []
wcss_scores = []

for k in ks:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df)
    score = silhouette_score(df, model.labels_)
    silhouette_scores.append(score)
    wcss_scores.append(model.inertia_)

plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
sb.lineplot(x=ks, y=silhouette_scores, color="red")
plt.title("Silhouette Score")
plt.grid(True)

plt.subplot(2, 1, 2)
sb.lineplot(x=ks, y=wcss_scores, color="blue")
plt.title("WCSS Score")
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nWe agree with k = 4 i.e. 4 clusters")

final_model = KMeans(n_clusters=4)
labels = final_model.fit_predict(df)

print(f"\nThe silhouette score is : {silhouette_score(df, labels)}")

df["CLUSTER_ID"] = labels

df.to_csv(path_or_buf="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/K_MEANS_RESULT.csv",
          index=False,
          header=True,
          sep=",")

# WE ARE SELECTING THIS ONE
