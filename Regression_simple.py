# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbrn
import statsmodels as stat 
from scipy import stats
#Construction de l'echantillon de training et de l'echantillon de test
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import sqrt


dtst = pd.read_csv('reg_simple.csv')

X1 = dtst.iloc[:,:-1].values
Y1 = dtst.iloc[:,-1].values

plt.scatter(X1,Y1)
plt.xlabel('heure_rev_independante_var')
plt.ylabel('note : dependante variable')
plt.style.use(['dark_background', 'fast'])
plt.show()


#Decouper notre dataset en données d'apprentissage(train) et données de test
#Créer des variables de training et test
X_train,X_test,y_train,y_test = train_test_split(X1,Y1, test_size= 0.2,random_state= 0 ) 
regresseur = LinearRegression()

#Monter regresseur par des variables
regresseur.fit(X_train,y_train)
#Etablir une prédiction en entrant de variables X_test et on compare les sorties avec y_test
y_prediction = regresseur.predict(X_test)



plt.scatter(X_test, y_test, color= 'red')
plt.xlabel('heure_rev_independante_var')
plt.ylabel('note : dependante variable')
plt.style.use(['dark_background', 'fast'])
plt.title('droite de regression')
#Créer droite de regression
plt.plot(X_train, regresseur.predict(X_train), color= 'blue')
plt.show()

#Faire prediction de 28 heure (x), combien va donne de note (y) ?
regresseur.predict(18)


#Calculer les écarts entre les prédictions et les observations (far9 bin var wa9i3 o var Prédiction)
#Créer fonction de calcule de l'ecart erreur
def rmse(z,z_hat):     # z=valeur observable  z_hat=valeur prédictive
    y_actual=np.array(z)
    y_pred=np.array(z_hat)
    error=(y_actual-y_pred)**2
    error_mean=round(np.mean(error))
    err_sq=sqrt(error_mean)
    return err_sq

#Pour tester
rmse(66, regresseur.predict(18))















