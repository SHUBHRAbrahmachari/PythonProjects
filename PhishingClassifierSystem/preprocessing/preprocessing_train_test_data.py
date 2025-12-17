import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
from pymongo import MongoClient
import pickle
import warnings

warnings.filterwarnings(action="ignore")

client = MongoClient(os.getenv("MONGODB_URL"))
db = client["PhishingClassifierProject"]
collection = db["raw_dataset"]

records = collection.find({})

df = pd.DataFrame(data=records)
df.drop(columns=["_id"], inplace=True)
print(df)

x = df.drop(columns=["Result"], inplace=False)
y = df["Result"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipeline = Pipeline(
    steps=[
        ("scaling", StandardScaler()),
        ("PCA", PCA(n_components=0.99))
    ],
    verbose=True
)

column_transformer = ColumnTransformer(
    transformers=[
        ("transformed", pipeline, x.columns)
    ],
    n_jobs=-1,
    verbose=True,
    remainder="passthrough"
)

x_train_processed = pd.DataFrame(data=column_transformer.fit_transform(x_train), columns=column_transformer.get_feature_names_out())
x_test_processed = pd.DataFrame(data=column_transformer.transform(x_test), columns=column_transformer.get_feature_names_out())

x_train_processed["Result"] = y_train.values
x_test_processed["Result"] = y_test.values

print(x_train_processed, '\n\n', x_test_processed, '\n\n')

training_records = x_train_processed.to_dict(orient="records")
testing_records = x_test_processed.to_dict(orient="records")

db["processed_train_data"].insert_many(training_records)
db["processed_test_data"].insert_many(testing_records)

with open("/transformers/column_transformer.pkl", "wb") as file:
    pickle.dump(column_transformer, file)

# db["processed_train_data"].delete_many({})
# db["processed_test_data"].delete_many({})

client.close()
