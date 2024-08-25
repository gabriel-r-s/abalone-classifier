import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils import shuffle
from scipy.stats import kruskal, mannwhitneyu

# Definindo os modelos e métodos de votação
MODELS = ["KNN", "DT", "NB", "SVM", "MLP", "MV", "SV"]
VOTING_METHODS = ["Soma", "Voto Majoritário", "Borda Count"]

# Função para embaralhar e dividir os dados
def data_shuffle(df_data):
    df_data = shuffle(df_data)
    x = df_data.drop(columns=["Class"])
    y = df_data["Class"]
    
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, stratify=y, random_state=42)
    x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    return x_train, x_validation, x_test, y_train, y_validation, y_test

# Função para treinar os modelos e avaliar
def train_models():
    data = pd.read_csv('abalone.data', header=None)
    data.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

    data['Class'] = pd.cut(data['Rings'], [0, 8, 10, float("inf")], labels=[0, 1, 2])
    data = pd.get_dummies(data, columns=["Sex"])

    acc_out = pd.DataFrame(columns=MODELS)
    acc_voting_out = pd.DataFrame(columns=VOTING_METHODS)
    
    for i in range(20):
        x_train, x_validation, x_test, y_train, y_validation, y_test = data_shuffle(data)
        
        # Treinamento dos modelos
        knn_model = KNeighborsClassifier(n_neighbors=7, weights='distance', p=2)
        dt_model = DecisionTreeClassifier(max_depth=15, min_samples_split=4, min_samples_leaf=3)
        nb_model = GaussianNB(var_smoothing=1e-8)
        svm_model = SVC(kernel='rbf', C=10, gamma=0.1, probability=True)
        mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)
        
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
        acc_out.loc[i] = [knn_acc, dt_acc, nb_acc, svm_acc, mlp_acc, majority_voting_acc, sum_voting_acc]
        acc_voting_out.loc[i] = [sum_voting_acc, majority_voting_acc, borda_count_acc]

    # Gravar os resultados em CSV
    acc_out.to_csv("acc_out_2011_exec.csv", index=False)
    acc_voting_out.to_csv("voting_methods_results11.csv", index=False)
    return acc_out, acc_voting_out

def load_models(path):
    acc_out = pd.read_csv(path)
    return acc_out

def eval_classifiers(acc_out):
    mean_acc = acc_out.mean()
    std_acc = acc_out.std()

    print("Média de Acurácia dos Modelos após 20 Execuções:")
    print(mean_acc)
    plt.bar(MODELS, mean_acc)
    plt.title("Média de acurácia")
    plt.ylim((0, 1))
    plt.show()

    print("\nDesvio Padrão das Acurácias:")
    print(std_acc)
    plt.bar(MODELS, std_acc)
    plt.title("Desvio padrão de acurácia")
    plt.ylim((0, 1))
    plt.show()

    print("\nGráfico de Dispersão das Acurácias:")
    for model in MODELS:
        plt.scatter([model] * len(acc_out), acc_out[model], label=model)
    plt.title("Dispersão das Acurácias dos Modelos")
    plt.xlabel("Modelos")
    plt.ylabel("Acurácia")
    plt.ylim((0, 1))
    plt.grid(True)
    plt.show()

    print("\nKruskal-Wallis Test para Classificadores:")
    stat, p_value = kruskal(*[acc_out[model] for model in MODELS if model in acc_out])
    print(f"Kruskal-Wallis H-statistic: {stat}, p-value: {p_value}")

    if p_value < 0.05:
        print("Há diferenças estatisticamente significativas entre os classificadores (p < 0.05)")
    else:
        print("Não há diferenças estatisticamente significativas entre os classificadores (p >= 0.05)")

    print("\nMann-Whitney U Test para Classificadores:")
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            stat, p_value = mannwhitneyu(acc_out[MODELS[i]], acc_out[MODELS[j]])
            print(f"{MODELS[i]} vs {MODELS[j]}: U-statistic: {stat}, p-value: {p_value}")

            if p_value < 0.05:
                print(f"Há diferenças estatisticamente significativas entre {MODELS[i]} e {MODELS[j]} (p < 0.05)")
            else:
                print(f"Não há diferenças estatisticamente significativas entre {MODELS[i]} e {MODELS[j]} (p >= 0.05)")

def eval_voting_methods(acc_voting_out):
    mean_acc_voting = acc_voting_out.mean()
    std_acc_voting = acc_voting_out.std()

    print("\nMédia de Acurácia dos Métodos de Votação:")
    print(mean_acc_voting)
    
    print("\nDesvio Padrão das Acurácias dos Métodos de Votação:")
    print(std_acc_voting)

    print("\nKruskal-Wallis Test para Métodos de Votação:")
    stat, p_value = kruskal(*[acc_voting_out[method] for method in VOTING_METHODS if method in acc_voting_out])
    print(f"Kruskal-Wallis H-statistic: {stat}, p-value: {p_value}")

    if p_value < 0.05:
        print("Há diferenças estatisticamente significativas entre os métodos de votação (p < 0.05)")
    else:
        print("Não há diferenças estatisticamente significativas entre os métodos de votação (p >= 0.05)")

    print("\nMann-Whitney U Test para Métodos de Votação:")
    for i in range(len(VOTING_METHODS)):
        for j in range(i + 1, len(VOTING_METHODS)):
            stat, p_value = mannwhitneyu(acc_voting_out[VOTING_METHODS[i]], acc_voting_out[VOTING_METHODS[j]])
            print(f"{VOTING_METHODS[i]} vs {VOTING_METHODS[j]}: U-statistic: {stat}, p-value: {p_value}")

            if p_value < 0.05:
                print(f"Há diferenças estatisticamente significativas entre {VOTING_METHODS[i]} e {VOTING_METHODS[j]} (p < 0.05)")
            else:
                print(f"Não há diferenças estatisticamente significativas entre {VOTING_METHODS[i]} e {VOTING_METHODS[j]} (p >= 0.05)")

if __name__ == "__main__":
    train = False
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg == "-t":
                train = True
    acc_out = None
    acc_voting_out = None
    if train:
        acc_out, acc_voting_out = train_models()
    else:
        acc_out = load_models("acc_out_2011_exec.csv")
        acc_voting_out = pd.read_csv("voting_methods_results11.csv")

    eval_classifiers(acc_out)
    eval_voting_methods(acc_voting_out)
