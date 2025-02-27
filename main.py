import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
data = pd.read_csv("data.csv", delimiter=";")  # Adjust delimiter if needed

# 2️⃣ Drop non-numeric columns
data = data.drop(columns=['id', 'glang'], errors='ignore')  # Drop categorical columns

# 3️⃣ Handle missing values
data = data.dropna()

# 4️⃣ Compute Correlation Matrix
correlation_matrix = data.corr()

# 5️⃣ Display Correlation Matrix
print("🔢 Correlation Matrix:\n", correlation_matrix)

# 6️⃣ Find Highly Correlated Variables (Above 0.7)
high_corr_pairs = []
threshold = 0.7  # Adjust threshold if needed

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

# 7️⃣ Display Highly Correlated Pairs
print("\n🔥 Highly Correlated Variables (|r| > 0.7):")
for var1, var2, corr_value in high_corr_pairs:
    print(f"{var1} ↔ {var2}  | Correlation: {corr_value:.2f}")

# 8️⃣ Heatmap Visualization
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("📊 Correlation Matrix Heatmap")
plt.show()
