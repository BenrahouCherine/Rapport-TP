
import numpy as np  
import matplotlib.pyplot as plt  


X = np.array([2, 4, 6, 8, 10, 12, 14, 16])  
Y = np.array([10, 25, 35, 40, 35, 25, 15, 5])  


XM = np.column_stack((np.ones(len(X)), X, X**2))


beta= np.linalg.inv(XM.T @ XM) @ (XM.T @ Y)

B0, B1, B2 = beta
print(f"B0 = {B0}, B1 = {B1}, B2 = {B2}")


Y_pred = B0 + B1*X + B2*(X**2)

a = np.sum((Y - Y_pred)**2)
b = np.sum((Y - np.mean(Y))**2)
R2 = 1 - a/b


print("R² =", round(R2, 4))


y_cont = B0 + B1*X + B2*(X**2)

plt.scatter(X, Y, color='blue', label='Données réelles')
plt.plot(X, y_cont, color='red', label='Régression polynomiale (degré 2)')
plt.title("Régression polynomiale ")
plt.xlabel("Prix du produit ")
plt.ylabel("Ventes ")
plt.legend()
plt.grid(True)
plt.show()