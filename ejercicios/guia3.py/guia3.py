import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Cargar el conjunto de datos Digits
def ejer1():
    digits = load_digits()
    X, y = digits.data, digits.target

    # Dividir el conjunto de datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configurar el MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, random_state=42)

    # Entrenar el modelo en el conjunto de entrenamiento
    x = X[0,:]
    mlp.fit(X_train, y_train)
    print(mlp.predict([X[4,:]]))

    # print(mlp.predict(X[0]))

    # Evaluar el rendimiento en el conjunto de prueba
    accuracy = mlp.score(X_test, y_test)

    # Mostrar la tasa de acierto
    print(f'Tasa de acierto en el conjunto de prueba: {accuracy:.2f}')


def ejer1_kfold(k=5, modelo="mlp"):
    digits = load_digits()
    X, y = digits.data, digits.target

    model = None

    if modelo =="mlp":
        model = MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, random_state=42)
    elif modelo == "nb":
        model = GaussianNB()
    elif modelo == "lda":
        model = LinearDiscriminantAnalysis()
    elif modelo == "kn":
        n_neighbors = 7
        model = KNeighborsClassifier(n_neighbors)
    elif modelo == "dt":
        model = DecisionTreeClassifier(random_state=0)
    elif modelo == "svm":
        model = svm.SVC()
    elif modelo == "bagging":
        model = BaggingClassifier()
    elif modelo == "adaboost":
        model = AdaBoostClassifier()
    else:
        raise "modelo no valido"
    # kf = KFold(n_splits=k)

    # porcentaje_acierto = []
    
    # print(kf)
    # for i, (train_index, test_index) in enumerate(kf.split(X)):
    #     X_train, y_train = X[train_index], y[train_index]
    #     X_test, y_test = X[test_index], y[test_index]

    #     # Entrenar el modelo en el conjunto de entrenamiento
    #     mlp.fit(X_train, y_train)

    #     # Evaluar el rendimiento en el conjunto de prueba
    #     accuracy = mlp.score(X_test, y_test)
    #     porcentaje_acierto.append(accuracy)
    #     print(f'Tasa de acierto en el k: {i} el conjunto de prueba: {accuracy:.2f}')
        # Validación cruzada con KFold
    
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = cross_val_score(model, X, y, cv=kfold)
    mean_accuracy = np.mean(accuracies)
    variance_accuracy = np.var(accuracies)
    print(f'{k} particiones - Mean Accuracy: {mean_accuracy:.2f}, Variance: {variance_accuracy:.4f}')
    
    # print(f'Media tasa de acierto: {np.mean(porcentaje_acierto)} desvio: {np.std(porcentaje_acierto)}')

ejer1_kfold(k=5, modelo="mlp")
ejer1_kfold(k=5, modelo="nb")
ejer1_kfold(k=5, modelo="lda")
ejer1_kfold(k=5, modelo="kn")
ejer1_kfold(k=5, modelo="dt")
ejer1_kfold(k=5, modelo="svm")
ejer1_kfold(k=5, modelo="bagging")
ejer1_kfold(k=5, modelo="adaboost")