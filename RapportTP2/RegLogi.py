import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

data_cancer = load_breast_cancer()
data = pd.DataFrame(data_cancer.data, columns=data_cancer.feature_names)
data['Outcome'] = data_cancer.target

print("Aperçu du dataset :")
print(data.head())



print("\nLe nombre de valeurs nulles par colonne :\n", data.isnull().sum())


for col in data.columns:
    if data[col].isnull().sum() > 0:
        med = data[col].median()
        data[col] = data[col].fillna(med)
        print(f"{col}: NaN remplacés par la médiane = {med}")

print("\nLe nombre de données dupliquées est :", data.duplicated().sum())
if data.duplicated().sum() > 0:
    data = data.drop_duplicates()


X = data.drop('Outcome', axis=1)
Y = data['Outcome']  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

acc_base = accuracy_score(Y_test, Y_pred)
print("\n=== Modèle de base ===")
print("Précision du modèle :", acc_base)
print("\nMatrice de confusion :\n", confusion_matrix(Y_test, Y_pred))
print("\nRapport de classification :\n", classification_report(Y_test, Y_pred))

model_l2 = LogisticRegression(penalty='l2', C=10, solver='lbfgs', max_iter=1000)
model_l2.fit(X_train, Y_train)
y_pred_L2 = model_l2.predict(X_test)

acc_l2 = accuracy_score(Y_test, y_pred_L2)
print("\n=== Régularisation L2 (Ridge) ===")
print("Précision :", acc_l2)
print("\nMatrice de confusion :\n", confusion_matrix(Y_test, y_pred_L2))
print("\nRapport de classification :\n", classification_report(Y_test, y_pred_L2))


model_L1 = LogisticRegression(penalty='l1', C=10, solver='liblinear', max_iter=1000)
model_L1.fit(X_train, Y_train)
y_pred_L1 = model_L1.predict(X_test)

acc_l1 = accuracy_score(Y_test, y_pred_L1)
print("\n=== Régularisation L1 (Lasso) ===")
print("Précision :", acc_l1)
print("\nMatrice de confusion :\n", confusion_matrix(Y_test, y_pred_L1))
print("\nRapport de classification :\n", classification_report(Y_test, y_pred_L1))


models = ['Logistique', 'L2 (Ridge)', 'L1 (Lasso)']
accuracies = [acc_base, acc_l2, acc_l1]

plt.figure(figsize=(7,5))
bars = plt.bar(models, accuracies, color=['blue','green','orange'])
plt.ylabel('Précision')
plt.ylim([0.9, 1.0])  
plt.title("Comparaison des précisions des modèles logistiques")
plt.grid(axis='y', linestyle='--', alpha=0.7)


for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f"{yval:.4f}", ha='center', va='bottom')

plt.show()





cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Malignant (0)', 'Benign (1)'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion — Modèle Logistique (de base)")
plt.grid(False)
plt.tight_layout()
plt.show()



cm_l1 = confusion_matrix(Y_test, y_pred_L1)
ConfusionMatrixDisplay(confusion_matrix=cm_l1,
                       display_labels=['Malignant (0)', 'Benign (1)']).plot(cmap=plt.cm.Oranges)
plt.title("Matrice de confusion — Logistique L1 (Lasso)")
plt.tight_layout()
plt.show()


cm_l2 = confusion_matrix(Y_test, y_pred_L2)
ConfusionMatrixDisplay(confusion_matrix=cm_l2,
                       display_labels=['Malignant (0)', 'Benign (1)']).plot(cmap=plt.cm.Greens)
plt.title("Matrice de confusion — Logistique L2 ")
plt.tight_layout()
plt.show()