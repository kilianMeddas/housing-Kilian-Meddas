import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


directory_name = "output"
try:
    os.mkdir(directory_name)
    print(f"Directory '{directory_name}' created successfully")
    print("See the result in it")
except FileExistsError:
    print(f"Directory '{directory_name}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{directory_name}'.")
except Exception as e:
    print(f"An error occurred: {e}")
print('\n\n\n')


df = pd.read_csv("housing.csv")
df.drop_duplicates()

print(f'Ten head of csv :  \n\n{df.head(10)}\n\n')
print(f"Describe of csv : \n\n{df.describe()}\n\n")
print(f"\n\n {df.info()} : We don't need to use fillna according to the df.info")

# Calculate the correlation matrix
corr_matrix = df.corr(numeric_only=True)
# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16)

# Save the heatmap
plt.savefig("output/heatmap.png", dpi=300, bbox_inches="tight")

print(f"\n\n\n\n\n See the results in '{directory_name}'")