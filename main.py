
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

file_path = Path("data/marketing_campaign.csv")
df = pd.read_csv(file_path, sep="\t")

df.loc[df["Income"] == 666666, "Income"] = None

age = 2026 - df["Year_Birth"]
df.loc[age > 100, "Year_Birth"] = None
df["Age"] = 2026 - df["Year_Birth"]

df["Income"] = df["Income"].fillna(df["Income"].mean())

features = [
    "Age", "Income", "Kidhome", "Teenhome", "Recency",
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
    "NumDealsPurchases", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth"
]

X = df[features].copy()
X["Age"] = X["Age"].fillna(X["Age"].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia_values = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia_values, marker="o")
plt.title("Elbow Method՝ կլաստերների քանակի ընտրություն")
plt.xlabel("Կլաստերների քանակը (k)")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=300)
plt.show()

print("Inertia արժեքները:")
for k, value in enumerate(inertia_values, start=1):
    print(f"k = {k}: {value:.2f}")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nՀաճախորդների բաշխումն ըստ խմբերի:")
print(df["Cluster"].value_counts().sort_index())

cluster_analysis = df.groupby("Cluster")[features].mean()

print("\nՅուրաքանչյուր խմբի միջին ցուցանիշները:")
print(cluster_analysis.round(2).T.to_string())
cluster_analysis["Income"].plot(kind="bar")

plt.title("Հաճախորդների միջին եկամուտն ըստ խմբերի")
plt.xlabel("Հաճախորդների խմբեր")
plt.ylabel("Միջին եկամուտ")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("cluster_income.png", dpi=300)
plt.show()
print("\nՀաճախորդների խմբերի միջին եկամուտները:")
print(cluster_analysis["Income"].round(2))

df.to_csv("customer_segments.csv", index=False)

print("\nՀաճախորդների խմբերը պահպանվել են customer_segments.csv ֆայլում:")