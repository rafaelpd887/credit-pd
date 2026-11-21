import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo


# ==================================================
# 1. LOAD DATASET
# ==================================================

default_of_credit_card_clients = fetch_ucirepo(id=350)

# Predictor variables
X = default_of_credit_card_clients.data.features

# Target variable
y = default_of_credit_card_clients.data.targets

# Rename predictor variables
column_mapping = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6"
}

X = X.rename(columns=column_mapping)

# Complete dataset for exploratory analysis
df = pd.concat([X, y], axis=1)

# Identify target column
target = y.columns[0]

pd.set_option("display.max_columns", None)


# ==================================================
# 2. INITIAL INSPECTION
# ==================================================

print("\nDataset shape:")
print(df.shape)

print("\nFirst rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe().T)


# ==================================================
# 3. DATASET METADATA
# ==================================================

print("\nDataset metadata:")
print(default_of_credit_card_clients.metadata)

print("\nVariable information:")
print(default_of_credit_card_clients.variables)


# ==================================================
# 4. TARGET VARIABLE ANALYSIS
# ==================================================

print("\nTarget column:")
print(target)

print("\nTarget frequency:")
print(df[target].value_counts())

print("\nTarget distribution (%):")
print(df[target].value_counts(normalize=True) * 100)

# Target visualization
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x=target)

plt.title("Distribution of Credit Default")
plt.xlabel("Default in the following month")
plt.ylabel("Number of clients")

plt.tight_layout()
plt.show()


# ==================================================
# 5. PREDICTOR DISTRIBUTIONS
# ==================================================

columns = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]

df[columns].hist(figsize=(12, 8), bins=30)

plt.suptitle("Distribution of Selected Predictor Variables")
plt.tight_layout()
plt.show()


# ==================================================
# 6. COMPARISON BY TARGET CLASS
# ==================================================

print("\nMean values by default status:")

print(
    df.groupby(target)[columns]
      .mean()
      .T
)


# ==================================================
# 7. CATEGORICAL VARIABLES ANALYSIS
# ==================================================

categorical_columns = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

# Category frequencies
for col in categorical_columns:

    print(f"\n{'=' * 40}")
    print(f"{col} - Category frequencies")
    print("=" * 40)

    print(
        df[col]
        .value_counts(dropna=False)
        .sort_index()
    )


# Default rate by category
for col in categorical_columns:

    print(f"\n{'=' * 40}")
    print(f"{col} - Default rate by category")
    print("=" * 40)

    result = (
        df.groupby(col, dropna=False)[target]
          .agg(["count", "mean"])
          .rename(columns={
              "count": "number_of_clients",
              "mean": "default_rate"
          })
    )

    result["default_rate"] *= 100

    print(result)