from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from preprocessing.numerical_preprocessor import NumericalColumnPreprocessor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import math
import warnings
import pickle

warnings.filterwarnings(action="ignore")

df = pd.read_csv(filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/raw_marketing_campaign.csv",
                 header=0,
                 sep=",").select_dtypes(exclude="object")

# `ID` column is of simple no use so we discard it at first
df.drop(columns=["ID"], inplace=True)
print(df.columns)

size = df.columns.size
lines = math.ceil(math.sqrt(size))

plt.figure(figsize=(15, 9))
for index, col in zip(range(1, size+1), df.columns):
    plt.subplot(lines, lines, index)
    sb.histplot(x=df[col], kde=True, bins=20, cumulative=False, discrete=False, palette="coolwarm")
    plt.title(col)
    plt.tight_layout()
    plt.grid(True)
plt.show()

# we see `Z_CostContact` and `Z_Revenue` has absolutely no variation. So we drop them as well
df.drop(columns=["Z_CostContact", "Z_Revenue"], inplace=True)
print(df.columns)

# now let us see if outliers are present or not
plt.figure(figsize=(15, 9))
for index, col in zip(range(1, size+1), df.columns):
    plt.subplot(lines, lines, index)
    sb.boxplot(x=df[col], palette="coolwarm")
    plt.title(col)
    plt.tight_layout()
    plt.grid(True)
plt.show()

# So we see almost all of them have outliers present in them.. so we need to fix it using Preprocessing

pipeline = Pipeline(
    steps=[
        ("NULL IMPUTATION WITH OUTLIER TREATMENT", NumericalColumnPreprocessor()),
        ("SCALING", StandardScaler()),
        ("PCA", PCA(n_components=0.99))
    ],
    verbose=True
)

column_transformer = ColumnTransformer(
    transformers=[
        ("PREPROCESSING", pipeline, df.columns)
    ],
    verbose=True,
    remainder="passthrough",
    n_jobs=4
)

df_transformed = pd.DataFrame(data=column_transformer.fit_transform(df), columns=column_transformer.get_feature_names_out())

with open("C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/eda/transformers/numerical_transformer.pkl", "wb") as file:
    pickle.dump(column_transformer, file)

print(df_transformed.duplicated().sum())

size = df_transformed.columns.size
lines = math.ceil(math.sqrt(size))

plt.figure(figsize=(15, 9))
for index, col in zip(range(1, size+1), df_transformed.columns):
    plt.subplot(lines, lines, index)
    sb.histplot(x=df_transformed[col], kde=True, bins=20, cumulative=False, discrete=False, palette="coolwarm")
    plt.title(col)
    plt.tight_layout()
    plt.grid(True)
plt.show()

df_transformed.to_csv(path_or_buf="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/marketing_campaign_numerical_processed.csv",
                      header=True,
                      index=False,
                      sep=",")


