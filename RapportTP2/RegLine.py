import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt 

# ==============================
# 1. Chargement du dataset
# ==============================
data = pd.read_csv("StudentsPerformance.csv")

print("Aperçu du dataset :")
print(data.head())

print("\nValeurs nulles par colonne :\n", data.isnull().sum())

# ==============================
# 2. Encodage des variables catégorielles
# ==============================
cat_features = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
data_encoded = pd.get_dummies(data, columns=cat_features, drop_first=True)

print("\nAperçu après encodage :")
print(data_encoded.head())

# ==============================
# 3. Séparation des variables
# ==============================
X = data_encoded.drop("math score", axis=1)
Y = data_encoded["math score"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ==============================
# 4. Normalisation
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 5. Régression linéaire simple
# ==============================
model = LinearRegression()
model.fit(X_train_scaled, Y_train)
Y_pred = model.predict(X_test_scaled)

mse_linear = mean_squared_error(Y_test, Y_pred)
r2_linear = r2_score(Y_test, Y_pred)

print("\n=== Régression Linéaire ===")
print(f"MSE : {mse_linear:.4f}")
print(f"R² : {r2_linear:.4f}")

# ==============================
# 6. Régression Ridge (L2)
# ==============================
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, Y_train)
Y_pred_ridge = ridge.predict(X_test_scaled)

mse_ridge = mean_squared_error(Y_test, Y_pred_ridge)
r2_ridge = r2_score(Y_test, Y_pred_ridge)

print("\n=== Régression Ridge (L2) ===")
print(f"MSE : {mse_ridge:.4f}")
print(f"R² : {r2_ridge:.4f}")

# ==============================
# 7. Régression Lasso (L1)
# ==============================
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, Y_train)
Y_pred_lasso = lasso.predict(X_test_scaled)

mse_lasso = mean_squared_error(Y_test, Y_pred_lasso)
r2_lasso = r2_score(Y_test, Y_pred_lasso)

print("\n=== Régression Lasso (L1) ===")
print(f"MSE : {mse_lasso:.4f}")
print(f"R² : {r2_lasso:.4f}")

# ==============================
# 8. Visualisation : Valeurs réelles vs prédites
# ==============================
plt.figure(figsize=(6,5))
plt.scatter(Y_test, Y_pred, color='blue', label='Prédictions linéaires')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', label='Idéal')
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Régression linéaire - valeurs réelles vs prédites")
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# 9. Comparaison des MSE entre modèles
# ==============================
models = ['Linéaire', 'Ridge (L2)', 'Lasso (L1)']
mse_values = [mse_linear, mse_ridge, mse_lasso]

plt.figure(figsize=(7,5))
plt.bar(models, mse_values, color=['blue','green','orange'])
plt.ylabel('Erreur quadratique moyenne (MSE)')
plt.title("Comparaison des erreurs MSE entre modèles")
plt.grid(axis='y')
plt.show()

# ==============================
# 10. Résumé global
# ==============================
print("\nRésumé comparatif :")
summary = pd.DataFrame({
    'Modèle': models,
    'MSE': mse_values,
    'R²': [r2_linear, r2_ridge, r2_lasso]
})
print(summary)