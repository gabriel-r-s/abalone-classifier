import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils import shuffle  # Corrigido aqui
from sklearn import metrics

# Carregar e preparar os dados como anteriormente
data = pd.read_csv('abalone.data', header=None)
data.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

# Agrupar classes de Rings
def classify_rings(rings):
    if rings <= 8:
        return 0
    elif rings <= 10:
        return 1
    else:
        return 2

data['Class'] = data['Rings'].apply(classify_rings)
data = data.drop(columns=["Rings"])
data = pd.get_dummies(data, columns=["Sex"])

print(data)

# Função para embaralhar os dados e separar em treino, validação, teste
def data_shufle(df_data):
    df_data = shuffle(df_data)  # Corrigido aqui
    x = df_data.drop(columns=["Class"])
    y = df_data["Class"]
    
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, stratify=y, random_state=42)
    x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    return x_train, x_validation, x_test, y_train, y_validation, y_test

# Loop para 20 execuções
acc_out = pd.DataFrame(columns=["KNN", "DT", "NB", "SVM", "MLP", "MV", "SV"])

for i in range(20):
    x_train, x_validation, x_test, y_train, y_validation, y_test = data_shufle(data)
    
    # Treinamento dos modelos
    knn_model = KNeighborsClassifier(n_neighbors=5)
    dt_model = DecisionTreeClassifier(max_depth=10)
    nb_model = GaussianNB()
    svm_model = SVC(kernel='rbf', C=1, probability=True)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=500)
    
    knn_model.fit(x_train, y_train)
    dt_model.fit(x_train, y_train)
    nb_model.fit(x_train, y_train)
    svm_model.fit(x_train, y_train)
    mlp_model.fit(x_train, y_train)
    
    # Avaliação dos modelos individuais
    knn_acc = knn_model.score(x_test, y_test)
    dt_acc = dt_model.score(x_test, y_test)
    nb_acc = nb_model.score(x_test, y_test)
    svm_acc = svm_model.score(x_test, y_test)
    mlp_acc = mlp_model.score(x_test, y_test)
    
    # Voting Classifiers
    estimators = [('KNN', knn_model), ('DT', dt_model), ('NB', nb_model), ('SVM', svm_model), ('MLP', mlp_model)]
    
    # Majority Voting
    majority_voting = VotingClassifier(estimators, voting='hard')
    majority_voting.fit(x_train, y_train)
    majority_voting_acc = majority_voting.score(x_test, y_test)
    
    # Sum Voting
    sum_voting = VotingClassifier(estimators, voting='soft')
    sum_voting.fit(x_train, y_train)
    sum_voting_acc = sum_voting.score(x_test, y_test)
    
    # Armazenar os resultados da execução
    acc_out.loc[i] = [knn_acc, dt_acc, nb_acc, svm_acc, mlp_acc, majority_voting_acc, sum_voting_acc]

# Calcular a média e o desvio padrão de cada classificador
mean_acc = acc_out.mean()
std_acc = acc_out.std()

print("Média de Acurácia dos Modelos após 20 Execuções:")
print(mean_acc)

print("\nDesvio Padrão das Acurácias:")
print(std_acc)

# Salvar os resultados em um arquivo CSV
acc_out.to_csv("acc_out_20_exec.csv", index=False)
