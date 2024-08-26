import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils import shuffle 
from sklearn import metrics
from scipy.stats import kruskal, mannwhitneyu

# Carregar e preparar os dados
data = pd.read_csv('abalone.data', header=None)
data.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

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

def data_shuffle(df_data):
    df_data = shuffle(df_data) 
    x = df_data.drop(columns=["Class"])
    y = df_data["Class"]
    
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, stratify=y, random_state=42)
    x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    return x_train, x_validation, x_test, y_train, y_validation, y_test

# Loop para 20 execuções
acc_out = pd.DataFrame(columns=["KNN", "DT", "NB", "SVM", "MLP"])
acc_voting_out = pd.DataFrame(columns=["Soma", "Voto Majoritário", "Borda Count"])

for i in range(20):
    x_train, x_validation, x_test, y_train, y_validation, y_test = data_shuffle(data)
    
    # Treinamento dos modelos
    knn_model = KNeighborsClassifier(n_neighbors=7, weights='distance', p=2)
    dt_model = DecisionTreeClassifier(max_depth=15, min_samples_split=4, min_samples_leaf=3)
    nb_model = GaussianNB()
    svm_model = SVC(kernel='rbf', C=10, gamma=0.1, probability=True)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500)
        
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
    
    # Borda Count
    probas = np.array([model.predict_proba(x_test) for _, model in estimators])
    borda_scores = probas.mean(axis=0)
    borda_count_acc = (borda_scores.argmax(axis=1) == y_test).mean()

    # Armazenar os resultados da execução
    acc_out.loc[i] = [knn_acc, dt_acc, nb_acc, svm_acc, mlp_acc]

    # Armazenar os resultados dos métodos de votação
    acc_voting_out.loc[i] = [sum_voting_acc, majority_voting_acc, borda_count_acc]  

# Calcular a média e o desvio padrão de cada classificador
mean_acc = acc_out.mean()
std_acc = acc_out.std()

print("Média de Acurácia dos Modelos após 20 Execuções:")
print(mean_acc)

print("\nDesvio Padrão das Acurácias:")
print(std_acc)

# Gravar os resultados dos métodos de votação em um arquivo CSV
acc_voting_out.to_csv("voting_methods_results.csv", index=False)

# Salvar os resultados individuais dos classificadores
acc_out.to_csv("acc_out_20_exec1.csv", index=False)

# Avaliar os classificadores
def eval_classifiers(acc_out):
    print("Acurácia dos classificadores")
    print(acc_out)

    print("Média de Acurácia dos Classificadores:")
    mean_acc_classifiers = acc_out.mean()
    print("\nDesvio Padrão das Acurácias dos Classificadores:")
    std_acc_classifiers = acc_out.std()

    # Teste de Kruskal-Wallis para classificadores
    print("\nKruskal-Wallis Test para Classificadores:")
    stat, p_value = kruskal(*[acc_out[col] for col in ["KNN", "DT", "NB", "SVM", "MLP"]])
    print(f"\tKruskal-Wallis H-statistic: {stat}, p-value: {p_value}")

    if p_value < 0.05:
        print("\t\tHá diferenças estatisticamente significativas entre os classificadores (p < 0.05)")
    else:
        print("\t\tNão há diferenças estatisticamente significativas entre os classificadores (p >= 0.05)")

    # Teste de Mann-Whitney U para todos os pares de classificadores
    print("\nMann-Whitney U Test para Classificadores:")

    classifiers = ["KNN", "DT", "NB", "SVM", "MLP"]
    for i in range(len(classifiers)):
        for j in range(i + 1, len(classifiers)):
            stat, p_value = mannwhitneyu(acc_out[classifiers[i]], acc_out[classifiers[j]])
            print(f"\t{classifiers[i]} vs {classifiers[j]}: U-statistic: {stat}, p-value: {p_value}")

            if p_value < 0.05:
                print(f"\t\tHá diferenças estatisticamente significativas entre {classifiers[i]} e {classifiers[j]} (p < 0.05)")
            else:
                print(f"\t\tNão há diferenças estatisticamente significativas entre {classifiers[i]} e {classifiers[j]} (p >= 0.05)")

# Avaliar os classificadores
eval_classifiers(acc_out)

# Avaliar os métodos de votação
def eval_voting_methods(acc_voting_out):
    print("Acurácia dos Métodos de Votação")
    print(acc_voting_out)

    mean_acc_voting = acc_voting_out.mean()
    std_acc_voting = acc_voting_out.std()

    print("\nMédia de Acurácia dos Métodos de Votação:")
    print(mean_acc_voting)
    
    print("\nDesvio Padrão das Acurácias dos Métodos de Votação:")
    print(std_acc_voting)

    # Teste de Kruskal-Wallis para métodos de votação
    print("\nKruskal-Wallis Test para Métodos de Votação:")
    stat, p_value = kruskal(*[acc_voting_out[col] for col in ["Soma", "Voto Majoritário", "Borda Count"]])
    print(f"\tKruskal-Wallis H-statistic: {stat}, p-value: {p_value}")

    if p_value < 0.05:
        print("\t\tHá diferenças estatisticamente significativas entre os métodos de votação (p < 0.05)")
    else:
        print("\t\tNão há diferenças estatisticamente significativas entre os métodos de votação (p >= 0.05)")

    # Teste de Mann-Whitney U para todos os pares de métodos de votação
    print("\nMann-Whitney U Test para Métodos de Votação:")

    voting_methods = ["Soma", "Voto Majoritário", "Borda Count"]
    for i in range(len(voting_methods)):
        for j in range(i + 1, len(voting_methods)):
            stat, p_value = mannwhitneyu(acc_voting_out[voting_methods[i]], acc_voting_out[voting_methods[j]])
            print(f"\t{voting_methods[i]} vs {voting_methods[j]}: U-statistic: {stat}, p-value: {p_value}")

            if p_value < 0.05:
                print(f"\t\tHá diferenças estatisticamente significativas entre {voting_methods[i]} e {voting_methods[j]} (p < 0.05)")
            else:
                print(f"\t\tNão há diferenças estatisticamente significativas entre {voting_methods[i]} e {voting_methods[j]} (p >= 0.05)")

# Avaliar os métodos de votação
eval_voting_methods(acc_voting_out)

def eval_best_of_each(acc_out, acc_voting_out):
    stat, p_value = mannwhitneyu(acc_out["MLP"], acc_voting_out["Voto Majoritário"])
    print(f"\tMLP vs Voto Majoritário: U-statistic: {stat}, p-value: {p_value}")

    if p_value < 0.05:
        print(f"\t\tHá diferenças estatisticamente significativas entre MLP e Voto Majoritário (p < 0.05)")
    else:
        print(f"\t\tNão há diferenças estatisticamente significativas entre MLP e Voto Majoritário (p >= 0.05)")
eval_best_of_each(acc_out, acc_voting_out)
