import math
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

df = pd.read_csv(filepath_or_buffer="C:/Users/Shubhra/OneDrive/Desktop/Datasets/marketing_campaign.csv",
                 sep="\t",
                 header=0).select_dtypes(include="object")


print(f"\nThe columns are : {df.columns}, let's analyse them one by one\n")

plt.figure(figsize=(15, 7))
for index, col in zip(range(1, 3), df.columns):
    plt.subplot(1, 2, index)
    sb.countplot(x=df[col])
    plt.tight_layout()
    plt.grid(True)
plt.show()

print(f"\nWe can't see `Dt_Customer` column\n")
print("\nMost of the people are either single or together or married\n")
print("\nMost of the people are from graduation and PhD level\n")

print(f"\n`Dt_Customer` is the joining date of a customer")

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
df["year_joined"] = df["Dt_Customer"].dt.year

print(df["Dt_Customer"])

"""
    Is the date when a customer joined in the business is much important for
    group prediction?

"""

plt.figure(figsize=(15, 9))
plt.title("customers joined")
sb.countplot(x=df["year_joined"])
plt.tight_layout()
plt.grid(True)
plt.show()
