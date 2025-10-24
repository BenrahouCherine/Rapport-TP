import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

#
X1 = np.array([4.0, 2.0, 3.0])
X2 = np.array([6.0, 5.0, 7.0])
Y = np.array([10.0, 7.0, 9.0])
n = len(Y)


X = np.column_stack((np.ones(n), X1, X2))


beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)
B0, B1, B2 = beta
print(f"Manuel → B0 = {B0:.4f}, B1 = {B1:.4f}, B2 = {B2:.4f}")


Y_pred = X @ beta


a = np.sum((Y - Y_pred)**2)   
b = np.sum((Y - np.mean(Y))**2)  
R2_manual = 1 - a/b
print("R² (manuel) =", round(R2_manual, 4))


RMSE = np.sqrt(np.mean((Y - Y_pred)**2))
print("RMSE =", round(RMSE, 4))


X_new = np.array([1, 3.5, 6.0])  
Y_new = X_new @ beta
print(f"Prédiction pour X1=3.5, X2=6.0 → Y = {Y_new:.4f}")


model = LinearRegression()
X_sklearn = np.column_stack((X1, X2))
model.fit(X_sklearn, Y)

print(f"Sklearn → B0 = {model.intercept_:.4f}, B1 = {model.coef_[0]:.4f}, B2 = {model.coef_[1]:.4f}")
R2_sklearn = model.score(X_sklearn, Y)
print("R² (Sklearn) =", round(R2_sklearn, 4))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X1, X2, Y, color='red', label='Données réelles')


X1_grid, X2_grid = np.meshgrid(np.linspace(min(X1), max(X1), 10),
                               np.linspace(min(X2), max(X2), 10))
Y_grid = B0 + B1 * X1_grid + B2 * X2_grid

ax.plot_surface(X1_grid, X2_grid, Y_grid, color='blue', alpha=0.5)

ax.set_xlabel('Heures de révision (X1)')
ax.set_ylabel('Heures de sommeil (X2)')
ax.set_zlabel('Note moyenne (Y)')
ax.set_title('Plan de régression linéaire multiple')
plt.legend()
plt.show()