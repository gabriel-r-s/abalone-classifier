import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils import shuffle
from scipy.stats import kruskal, mannwhitneyu

MODELS = ["KNN", "DT", "NB", "SVM", "MLP", "MV", "SV"]

def data_shufle(df_data):
    df_data = shuffle(df_data)
    x = df_data.drop(columns=["Class"])
    y = df_data["Class"]
    
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.5, stratify=y, random_state=42)
    x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    return x_train, x_validation, x_test, y_train, y_validation, y_test

def train_models():
    data = pd.read_csv('abalone.data', header=None)
    data.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

    data['Class'] = pd.cut(data['Rings'], [0, 8, 10, float("inf")], labels=[0, 1, 2])
    data = pd.get_dummies(data, columns=["Sex"])

    acc_out = pd.DataFrame(columns=MODELS)
    for i in range(20):
        x_train, x_validation, x_test, y_train, y_validation, y_test = data_shufle(data)
        
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
        
        knn_acc = knn_model.score(x_test, y_test)
        dt_acc = dt_model.score(x_test, y_test)
        nb_acc = nb_model.score(x_test, y_test)
        svm_acc = svm_model.score(x_test, y_test)
        mlp_acc = mlp_model.score(x_test, y_test)
        
        estimators = [('KNN', knn_model), ('DT', dt_model), ('NB', nb_model), ('SVM', svm_model), ('MLP', mlp_model)]
        
        majority_voting = VotingClassifier(estimators, voting='hard')
        majority_voting.fit(x_train, y_train)
        majority_voting_acc = majority_voting.score(x_test, y_test)
        
        sum_voting = VotingClassifier(estimators, voting='soft')
        sum_voting.fit(x_train, y_train)
        sum_voting_acc = sum_voting.score(x_test, y_test)

        acc_out.loc[i] = [knn_acc, dt_acc, nb_acc, svm_acc, mlp_acc, majority_voting_acc, sum_voting_acc]

    acc_out.to_csv("acc_out_20_exec.csv", index=False)
    return acc_out

def load_models(path):
    acc_out = pd.read_csv(path)
    return acc_out

def eval_models(acc_out):
    mean_acc = acc_out.mean()
    std_acc = acc_out.std()

    print("Média de Acurácia dos Modelos após 20 Execuções:")
    print(mean_acc)
    fig_acc, (ax_acc_mean, ax_acc_std) = plt.subplots(2)
    ax_acc_mean.bar(MODELS, mean_acc)
    ax_acc_mean.set_title("Média de acurácia")
    ax_acc_mean.set_ylim((0, 1))

    print("\nDesvio Padrão das Acurácias:")
    print(std_acc)
    ax_acc_std.bar(MODELS, mean_acc)
    ax_acc_std.set_title("Desvio padrão de acurácia")
    ax_acc_std.set_ylim((0, 1))


    # Gráfico de Dispersão
    fig_disp, ax_disp = plt.subplots()
    print("\nGráfico de Dispersão das Acurácias:")
    for model in MODELS:
        ax_disp.scatter([model] * len(acc_out), acc_out[model], label=model)

    ax_disp.set_title("Dispersão das Acurácias dos Modelos")
    ax_disp.set_xlabel("Modelos")
    ax_disp.set_ylabel("Acurácia")
    ax_disp.set_ylim((0, 1))
    ax_disp.grid(True)


    # Teste de Kruskal-Wallis
    print("\nKruskal-Wallis Test:")
    stat, p_value = kruskal(*[acc_out[model] for model in MODELS])
    print(f"\tKruskal-Wallis H-statistic: {stat}, p-value: {p_value}")

    if p_value < 0.05:
        print("\tHá diferenças estatisticamente significativas entre os modelos (p < 0.05)")
    else:
        print("\tNão há diferenças estatisticamente significativas entre os modelos (p >= 0.05)")

    # Teste de Mann-Whitney U para todos os pares de modelos
    print("\nMann-Whitney U Test:")

    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            stat, p_value = mannwhitneyu(acc_out[MODELS[i]], acc_out[MODELS[j]])
            print(f"\t{MODELS[i]} vs {MODELS[j]}: U-statistic: {stat}, p-value: {p_value}")

            if p_value < 0.05:
                print(f"\tHá diferenças estatisticamente significativas entre {MODELS[i]} e {MODELS[j]} (p < 0.05)")
            else:
                print(f"\tNão há diferenças estatisticamente significativas entre {MODELS[i]} e {MODELS[j]} (p >= 0.05)")

if __name__ == "__main__":
    train = False
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg == "-t":
                train = True
    acc_out = None
    if train:
        acc_out = train_models()
    else:
        acc_out = load_models("acc_out_20_exec.csv")

    eval_models(acc_out)

    try:
        if input("Show graphs? (y/N) ").lower().strip() == "y":
            plt.show()
    except (EOFError, KeyboardInterrupt):
        pass
