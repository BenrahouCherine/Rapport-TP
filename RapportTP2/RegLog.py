import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier


data = pd.read_csv ("iris.csv")
print(data.head())

X = data.drop ('variety' , axis= 1)
Y = data['variety']
data_shuffled = data.sample(frac=1, random_state=42)

print(data_shuffled.head())
X_train , X_test , y_train ,y_test = train_test_split(X , Y , test_size=0.2 , random_state=42 , stratify=Y )

model = DecisionTreeClassifier ()

model.fit(X_train,y_train)

Y_pre = model.predict(X_test)

acc1 = accuracy_score(y_test , Y_pre) 
print("\n Arbre de decision a pour accuracy ", acc1)


model2 = RandomForestClassifier (max_depth=3)
model2.fit (X_train , y_train)

y_pre2 = model2.predict(X_test)

acc2 = accuracy_score(y_test , y_pre2)
print("random forest a pour accuracy\n", acc2)

model_gb = GradientBoostingClassifier(
)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)

print("boosting accuracy \n",acc_gb)


model_boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42

)

model_boosting.fit(X_train,y_train)
y_pred_ad = model_boosting.predict(X_test)
acc_ad = accuracy_score(y_test,y_pred_ad)
print ("adaboost accuracy \n",acc_ad)



model_bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
    n_estimators=50,
    random_state=42,
    
)


model_bagging.fit(X_train, y_train)
y_pred_bag = model_bagging.predict(X_test)
acc_bag = accuracy_score(y_test, y_pred_bag)
print ("bagging accuracy \n",acc_bag)






##print("Arbre de decison \n")
##print(classification_report(y_test,Y_pre))

##print("Random forest \n")
##print(classification_report(y_test,y_pre2))

