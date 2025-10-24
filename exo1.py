import numpy as np
import matplotlib.pyplot as plt

X = np.array([420, 380, 350, 400, 440, 380, 450, 420])
Y = np.array([5.5, 6.0, 6.5, 6.0, 5.0, 6.5, 4.5, 5.0])



print("---- MÉTHODE 1 : Formules analytiques ----")

b1 = ((1/len(X)) * np.sum(X * Y)) - (np.mean(X) * np.mean(Y))
b2 = ((1/len(X)) * np.sum(X ** 2)) - (np.mean(X) ** 2)
B1 = b1 / b2
B0 = np.mean(Y) - (B1 * np.mean(X))

print(f"La droite de régression est : y = {B1:.4f}x + {B0:.4f}")


n = float(input("Saisir une valeur de X : "))
Yn = B1 * n + B0
print(f"La valeur prédite de Y pour X = {n} est : {Yn:.2f}")


plt.figure(figsize=(8,5))
plt.scatter(X, Y, color="red", label="Données réelles")
plt.plot(X, B1 * X + B0, color="blue", label="Droite de régression")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Régression linéaire simple — Méthode analytique")
plt.legend()
plt.grid(True)
plt.text(min(X), max(Y), f"y = {B1:.2f}x + {B0:.2f}", color="blue")
plt.show()



print("\n---- MÉTHODE 2 : Descente de gradient ----")


XN = (X - np.mean(X)) / np.std(X)
B0, B1 = 0, 0
nbIter = 1000
AL = 0.01
cost = []

for i in range(nbIter):
    JB0 = -(1/len(X)) * np.sum(Y - (B1 * XN + B0))
    JB1 = -(1/len(X)) * np.sum((Y - (B1 * XN + B0)) * XN)
    B0 -= AL * JB0
    B1 -= AL * JB1
    J = (1/(2*len(X))) * np.sum((Y - (B1 * XN + B0))**2)
    cost.append(J)

print(f"Après normalisation : B0 = {B0:.4f}, B1 = {B1:.4f}")

B1F = B1 / np.std(X)
B0F = B0 - (B1 * np.mean(X)) / np.std(X)
print(f"Coefficients sur les données initiales : B0 = {B0F:.4f}, B1 = {B1F:.4f}")


X_test = np.array([360, 380, 400, 420, 440, 450])
Y_test = np.array([6.6, 6.04, 5.73, 5.4, 4.84, 4.6])

Y_pred = B0F + B1F * X_test

RSS = np.sum((Y_test - Y_pred)**2)
R2 = 1 - ((np.sum((Y_test - Y_pred)**2)) / np.sum((Y_test - np.mean(Y_test))**2))
print(f"\nRSS = {RSS:.4f}")
print(f"R² = {R2:.4f}")


plt.figure(figsize=(8,5))
plt.scatter(XN, Y, color='blue')
plt.plot(XN, B1 * XN + B0, color='red')
plt.title("Régression linéaire — Descente de gradient")
plt.xlabel("X normalisé")
plt.ylabel("Y")
plt.grid(True)
plt.show()


plt.figure(figsize=(8,5))
plt.plot(cost)
plt.title("Évolution de la fonction de coût J")
plt.xlabel("Itérations")
plt.ylabel("J(B0, B1)")
plt.grid(True)
plt.show()


plt.figure(figsize=(8,5))
plt.scatter(X_test, Y_test, color="blue", label="Données réelles (test)")
plt.plot(X_test, Y_pred, color="red", label="Prédictions du modèle")
plt.legend()
plt.title("Évaluation du modèle sur un nouveau dataset")
plt.xlabel("X (test)")
plt.ylabel("Y (réel / prédit)")
plt.grid(True)
plt.show()