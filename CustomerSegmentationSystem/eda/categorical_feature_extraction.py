import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from preprocessing.categorical_preprocessor import CategoricalColumnPreprocessor
from preprocessing.numerical_preprocessor import NumericalColumnPreprocessor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sb
import warnings
import pickle

warnings.filterwarnings(action="ignore")

df = pd.read_csv(
    filepath_or_buffer="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/raw_marketing_campaign.csv",
    header=0,
    sep=",").select_dtypes(include="object")

print(df.info())

# we have only 3 categorical columns

for col in df.columns:
    print(df[col].unique())

# only Dt_Customer has so many values... that indicate when did the customer join

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
df["Customer_Duration"] = datetime.now() - df["Dt_Customer"]
df["Customer_Duration_Days"] = df["Customer_Duration"].dt.days
df.drop(columns=["Dt_Customer", "Customer_Duration"], inplace=True)

# `Customer_Duration_Days` is already integer

print(df.info())  # we see one numerical column there
df_categorical_columns = df.select_dtypes(include="object").columns

categorical_pipeline = Pipeline(
    steps=[
        ("NULL VALUE IMPUTATION AND ENCODING", CategoricalColumnPreprocessor()),
        ("SCALING", StandardScaler()),
        ("PCA", PCA(n_components=0.99))
    ],
    verbose=True
)

numerical_pipeline = Pipeline(
    steps=[
        ("NULL VALUE IMPUTATION AND OUTLIER TREATMENT", NumericalColumnPreprocessor()),
        ("SCALING", StandardScaler()),
        ("PCA", PCA(n_components=0.99))
    ],
    verbose=True
)

column_transformer = ColumnTransformer(
    transformers=[
        ("CATEGORICAL PREPROCESSING", categorical_pipeline, df_categorical_columns),
        ("NUMERICAL PREPROCESSING", numerical_pipeline, ["Customer_Duration_Days"])
    ],
    n_jobs=-1,
    remainder="passthrough",
    verbose=True
)

df_transformed = pd.DataFrame(data=column_transformer.fit_transform(df),
                              columns=column_transformer.get_feature_names_out())

with open("C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/eda/transformers/categorical_transformer.pkl", "wb") as file:
    pickle.dump(column_transformer, file)

plt.figure(figsize=(15, 9))
for index, col in zip(range(1, 4), df_transformed.columns):
    plt.subplot(1, 3, index)
    sb.histplot(x=df_transformed[col], bins=20, palette="muted", kde=True)
    plt.title(col)
    plt.tight_layout()
    plt.grid(True)

plt.show()

df_transformed.to_csv(
    path_or_buf="C:/Users/Shubhra/PycharmProjects/CustomerSegmentationSystem/dataset/marketing_campaign_categorical_processed.csv",
    header=True,
    index=False,
    sep=",")
