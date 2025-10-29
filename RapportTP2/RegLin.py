# ==========================
# 1. Importation des bibliothèques
# ==========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# ==========================
# 2. Chargement du dataset
# ==========================
data = pd.read_csv("StudentsPerformance.csv")

print("Aperçu du dataset :")
print(data.head())

# ==========================
# 3. Vérification des valeurs manquantes
# ==========================
print("\nValeurs nulles par colonne :\n", data.isnull().sum())

# ==========================
# 4. Encodage des variables catégorielles
# ==========================
# Colonnes catégorielles à encoder
cat_features = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

# Encodage One-Hot (conversion en variables numériques)
data_encoded = pd.get_dummies(data, columns=cat_features, drop_first=True)

print("\nAperçu après encodage :")
print(data_encoded.head())

# ==========================
# 5. Séparation des variables X et Y
# ==========================
X = data_encoded.drop("math score", axis=1)
Y = data_encoded["math score"]

# ==========================
# 6. Division du dataset
# ==========================
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ==========================
# 7. Normalisation
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# 8. Régression linéaire simple
# ==========================
model = LinearRegression()
model.fit(X_train_scaled, Y_train)
Y_pred = model.predict(X_test_scaled)

print("\n=== Régression Linéaire ===")
print("MSE :", mean_squared_error(Y_test, Y_pred))
print("R² :", r2_score(Y_test, Y_pred))

# ==========================
# 9. Régression Ridge (Régularisation L2)
# ==========================
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, Y_train)
Y_pred_ridge = ridge.predict(X_test_scaled)

print("\n=== Régression Ridge (L2) ===")
print("MSE :", mean_squared_error(Y_test, Y_pred_ridge))
print("R² :", r2_score(Y_test, Y_pred_ridge))

# ==========================
# 10. Régression Lasso (Régularisation L1)
# ==========================
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, Y_train)
Y_pred_lasso = lasso.predict(X_test_scaled)

print("\n=== Régression Lasso (L1) ===")
print("MSE :", mean_squared_error(Y_test, Y_pred_lasso))
print("R² :", r2_score(Y_test, Y_pred_lasso))