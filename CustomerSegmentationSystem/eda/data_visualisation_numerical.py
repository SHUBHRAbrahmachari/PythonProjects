import math
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import warnings

from preprocessing.numerical_preprocessor import NumericalColumnPreprocessor

warnings.filterwarnings(action="ignore")

df = pd.read_csv(filepath_or_buffer="C:/Users/Shubhra/OneDrive/Desktop/Datasets/marketing_campaign.csv",
                 header=0,
                 sep="\t")

# select the numerical columns

df_numerical = df.select_dtypes(exclude="object")

print(f"\nThe numerical columns are : {df_numerical.columns}\n")
print("\n`ID` literally doesn't make any sense so we just drop it")
df_numerical.drop(columns=["ID"], inplace=True)
print(f"\nFiltered numerical columns are : {df_numerical.columns}\n")

numerical_size = df_numerical.columns.size
size = math.ceil(math.sqrt(numerical_size))

plt.figure(figsize=(15, 9))
for index, col in zip(range(1, numerical_size+1), df_numerical.columns):
    plt.subplot(size, size, index)
    sb.histplot(x=df_numerical[col], palette="viridis", kde=True, bins=20)
    plt.title(f"{col}")
    plt.tight_layout()
    plt.grid(True)
plt.show()

print(f"\nWe see that (1) `Z_CostContact`, and (2) `Z_Revenue` are just constant columns. No variation so actually useless! so we drop them")
df_numerical.drop(columns=["Z_CostContact", "Z_Revenue"], inplace=True)
print(f"\nFiltered numerical columns are : {df_numerical.columns}\n")

print(f"\nNow let us see the violin plot/count plot to understand the variation\n")

numerical_size = df_numerical.columns.size
size = math.ceil(math.sqrt(numerical_size))

plt.figure(figsize=(15, 9))
for index, col in zip(range(1, numerical_size+1), df_numerical.columns):
    plt.subplot(size, size, index)
    plt.title(f"{col}")
    if df_numerical[col].unique().size >= 10:
        sb.violinplot(x=df_numerical[col], palette="viridis")
    else:
        sb.countplot(x=df_numerical[col], palette="viridis")
    plt.tight_layout()
    plt.grid(True)
plt.show()

print(f"\nMaximum user base is born in between 1959 and 1979, Baby Boomers and GenX\n")
print(f"\nMaximum salary ranges lie in between 5800 and 94900\n")
print(f"\nMost of the customers have 0 or 1 kids at home... very few have 2.. that too teenagers\n")
print(f"\nRecency is uniformly distributed\n")
print(f"\nMntWines in range 1 to 145 has the highest density\n")
print(f"\nMntFruits in range 1 to 20 has the highest density\n")
print(f"\nMntMeatProducts in range 1 to 140 has the highest density\n")
print(f"\nMntFishProducts in range 1 to 28 has the highest density\n")
print(f"\nMntSweetProducts in range 1 to 20 has the highest density\n")
print(f"\nMntGoldProds in range 1 to 40 has the highest density\n")
print(f"\nNumDealsPurchases in range 1 to 3 has the highest density\n")
print(f"\nNumWebPurchases in range 1 to 7 has the highest density\n")
print(f"\nNumCatalogPurchases in range 1 to 4 has the highest density\n")
print(f"\nNumStorePurchases in range 2 to 5 has the highest density\n")
print(f"\nNumWebVisitsMonth in range 5 to 8 has the highest density\n")
print(f"\nAcceptedCmp1/2/3/4/5, Complain, Response almost all are 0 except few 1s... Almost no variation in AcceptedCmp2, Complain\n")

print(f"\nNow let's see whether we see outliers or not\n")

plt.figure(figsize=(15, 9))
for index, col in zip(range(1, numerical_size+1), df_numerical.columns):
    plt.subplot(size, size, index)
    plt.title(f"{col}")
    sb.boxplot(x=df_numerical[col])
    plt.tight_layout()
    plt.grid(True)
plt.show()

df_numerical = pd.DataFrame(data=NumericalColumnPreprocessor().fit_transform(df_numerical))

plt.figure(figsize=(15, 9))
for index, col in zip(range(1, numerical_size+1), df_numerical.columns):
    plt.subplot(size, size, index)
    plt.title(f"{col}")
    sb.boxplot(x=df_numerical[col])
    plt.tight_layout()
    plt.grid(True)
plt.show()
