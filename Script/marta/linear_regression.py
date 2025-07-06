from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

bacteria_names = ['Acinetobacter_baumannii', 'Campylobacter_jejuni', 'Escherichia_coli', 'Neisseria_gonorrhoeae', 'Pseudomonas_aeruginosa', 'Salmonella_enterica','Staphylococcus_aureus', 'Streptococcus_pneumoniae', 'Klebsiella_pneumoniae']
for bacteria_file in bacteria_names:
    data = pd.read_csv(f"/srv/scratch/AMR/Reduced_genotype/{bacteria_file}_reduced_genotype.tsv")
      # e.g., columns: ['feature1', 'feature2', ..., 'target']

    X = data.drop('target', axis=1)  # Features
    y = data['target']               # Target variable (e.g., MIC values)

    # Split into training and test sets (you can adjust test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression()

    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print({{bacteria_file}})
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")