import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Define the data (only MCC now)
data = {
    "Species": [
        "Acinetobacter_baumannii",
        "Campylobacter_jejuni",
        "Escherichia_coli",
        "Neisseria_gonorrhoeae",
        "Pseudomonas_aeruginosa",
        "Salmonella_enterica",
        "Staphylococcus_aureus",
        "Streptococcus_pneumoniae",
        "Klebsiella_pneumoniae"
    ],
    "MCC": [0.564, 0.931, 0.377, 0.286, -0.039, 0.896, 0.793, 0.927, 0.586]
}

# Step 2: Create DataFrame
df = pd.DataFrame(data)
df.set_index("Species", inplace=True)

# Step 3: Plot (pastel purple color for a clean aesthetic)
fig, ax = plt.subplots(figsize=(12, 6))
df.plot(kind="bar", ax=ax, color="#CBAACB")  # Lavender pastel

# Step 4: Customize
plt.title("MCC by Bacterial Species", fontsize=14)
plt.ylabel("Matthews Correlation Coefficient (MCC)")
plt.xticks(rotation=45, ha='right')
plt.ylim(min(df["MCC"].min() - 0.1, -0.1), 1.05)  # to show negative bar if any
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend().remove()
plt.tight_layout()

# Step 5: Save and Show
plt.savefig("MCC_by_species_pastel.png", dpi=300, bbox_inches='tight')
plt.show()
