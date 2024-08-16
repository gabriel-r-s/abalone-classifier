#!/usr/bin/env python
# coding: utf-8


# In[1]:


import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB


# In[2]:


# dados = pd.read_csv("Vertebral.csv")
# dados = pd.read_csv("Banana.csv")
dados = pd.read_csv(
    "abalone.data",
    delimiter=",",
    header=None,
    converters={
        0: lambda s: {"M": 0.0, "F": 1.0}.get(s, 2.0),
        8: lambda r: 1 if int(r) <= 8 else 2 if int(r) <= 10 else 2,
    },
)
dados = shuffle(dados)
X = dados.iloc[:,:-1]
Y = dados.iloc[:,-1]


# **Gerando os conjuntos de treino, teste e validação**
# 

# 

# In[ ]:


x_treino,x_temp,y_treino,y_temp=train_test_split(X,Y,test_size=0.5,stratify=Y)
x_validacao,x_teste,y_validacao,y_teste=train_test_split(x_temp,y_temp,test_size=0.5, stratify = y_temp)

print("Treino")
x_treino.info()
y_treino.info()

print("\nValidação")
x_validacao.info()
y_validacao.info()

print("\nTeste")
x_teste.info()
y_teste.info()


# In[ ]:


from sklearn import metrics
def plot_roc_curve(fper,tper,cor,classsificador):
    plt.plot(fper, tper, color=cor, label=classsificador)
    plt.plot([0, 1], [0, 1], color="green", linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend()

# In[ ]:


#atribuindo valores aos hiperparâmetros
#n_neighbors corresponde ao tamanho da vizinhança
#weights indica se os vizinhos terão pesos diferentes ou não. Pode assumir os valores uniform ou distante (ou callabe)

KNN = KNeighborsClassifier(n_neighbors=13,weights="distance")
KNN.fit(x_treino,y_treino)
opiniao = KNN.predict(x_teste)

print("Desempenho KNN")
print("Matriz de Confusão\n ",confusion_matrix(y_teste,opiniao))
TN, FP, FN, TP = confusion_matrix(y_teste,opiniao).ravel()
print("TP: ", TP, " FN: ",FN," FP: ",FP," TN: ",TN)
print("Acurácia: ",accuracy_score(y_teste, opiniao))
print("Sensibilidade: ",(TP/(TP+FN)))
print("Especificade: ",(TN/(FP+TN)))
print("AUC: ",roc_auc_score(y_teste,opiniao))
print("F-Score: ",f1_score(y_teste, opiniao))
print("Precision: ",precision_score(y_teste, opiniao))
print("Recall: ",recall_score(y_teste, opiniao))
print("\n\n")

#print(KNN.predict_proba(x_teste))

y_score = KNN.predict_proba(x_teste)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_teste,y_score,pos_label=2)
plot_roc_curve(fpr, tpr,"red","KNN")


NB = GaussianNB()
NB.fit(x_treino,y_treino)
opiniao = NB.predict(x_teste)

print("Desempenho Naive Bayes")
print("Matriz de Confusão\n ",confusion_matrix(y_teste,opiniao))
TN, FP, FN, TP = confusion_matrix(y_teste,opiniao).ravel()
print("TP: ", TP, " FN: ",FN," FP: ",FP," TN: ",TN)
print("Acurácia: ",accuracy_score(y_teste, opiniao))
print("Sensibilidade: ",(TP/(TP+FN)))
print("Especificade: ",(TN/(FP+TN)))
print("AUC: ",roc_auc_score(y_teste,opiniao))
print("F-Score: ",f1_score(y_teste, opiniao))
print("Precision: ",precision_score(y_teste, opiniao))
print("Recall: ",recall_score(y_teste, opiniao))
print("\n\n")

y_score = NB.predict_proba(x_teste)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_teste,y_score,pos_label=2)
plot_roc_curve(fpr, tpr,"blue","Naïve Bayes")



DT = tree.DecisionTreeClassifier()
DT.fit(x_treino,y_treino)
opiniao = DT.predict(x_teste)

print("Desempenho Decision Tree")
print("Matriz de Confusão\n ",confusion_matrix(y_teste,opiniao))
TN, FP, FN, TP = confusion_matrix(y_teste,opiniao).ravel()
print("TP: ", TP, " FN: ",FN," FP: ",FP," TN: ",TN)
print("Acurácia: ",accuracy_score(y_teste, opiniao))
print("Sensibilidade: ",(TP/(TP+FN)))
print("Especificade: ",(TN/(FP+TN)))
print("AUC: ",roc_auc_score(y_teste,opiniao))
print("F-Score: ",f1_score(y_teste, opiniao))
print("Precision: ",precision_score(y_teste, opiniao))
print("Recall: ",recall_score(y_teste, opiniao))
print("\n\n")

y_score = DT.predict_proba(x_teste)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_teste,y_score,pos_label=2)
plot_roc_curve(fpr, tpr,"black","DT")

MLP = MLPClassifier(hidden_layer_sizes=(10,10),activation="relu",max_iter=500)
MLP.fit(x_treino,y_treino)
opiniao = MLP.predict(x_teste)


print("Desempenho Multilayer Perceptron")
print("Matriz de Confusão\n ",confusion_matrix(y_teste,opiniao))
TN, FP, FN, TP = confusion_matrix(y_teste,opiniao).ravel()
print("TP: ", TP, " FN: ",FN," FP: ",FP," TN: ",TN)
print("Acurácia: ",accuracy_score(y_teste, opiniao))
print("Sensibilidade: ",(TP/(TP+FN)))
print("Especificade: ",(TN/(FP+TN)))
print("AUC: ",roc_auc_score(y_teste,opiniao))
print("F-Score: ",f1_score(y_teste, opiniao))
print("Precision: ",precision_score(y_teste, opiniao))
print("Recall: ",recall_score(y_teste, opiniao))
print("\n\n")

y_score = MLP.predict_proba(x_teste)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_teste,y_score,pos_label=2)
plot_roc_curve(fpr, tpr,"green","MLP")




SVM = SVC(kernel='rbf',C=0.9,probability=True)
SVM.fit(x_treino,y_treino)
opiniao = SVM.predict(x_teste)

print("Desempenho SVM")
print("Matriz de Confusão\n ",confusion_matrix(y_teste,opiniao))
TN, FP, FN, TP = confusion_matrix(y_teste,opiniao).ravel()
print("TP: ", TP, " FN: ",FN," FP: ",FP," TN: ",TN)
print("Acurácia: ",accuracy_score(y_teste, opiniao))
print("Sensibilidade: ",(TP/(TP+FN)))
print("Especificade: ",(TN/(FP+TN)))
print("AUC: ",roc_auc_score(y_teste,opiniao))
print("F-Score: ",f1_score(y_teste, opiniao))
print("Precision: ",precision_score(y_teste, opiniao))
print("Recall: ",recall_score(y_teste, opiniao))
print("\n\n")

y_score = SVM.predict_proba(x_teste)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_teste,y_score,pos_label=2)
plot_roc_curve(fpr, tpr,"yellow","SVM")
plt.show()

