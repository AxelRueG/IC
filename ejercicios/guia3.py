import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits, load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# obtener modelo
def get_modelo(type_model="mlp", n_neighbors=3, rs=0, pb=False):
    
    if type_model =="mlp":
        # return MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, random_state=rs)
        return MLPClassifier(max_iter=1000, random_state=rs)
    elif type_model == "nb":
        return GaussianNB()
    elif type_model == "lda":
        return LinearDiscriminantAnalysis()
    elif type_model == "kn":
        return KNeighborsClassifier(n_neighbors)
    elif type_model == "dt":
        return DecisionTreeClassifier(random_state=rs)
    elif type_model == "svm":
        return svm.SVC(probability=pb, kernel="rbf")
    else:
        raise "modelo no valido"


# Cargar el conjunto de datos Digits
def ejer1():
    digits = load_digits()
    X, y = digits.data, digits.target

    # Dividir el conjunto de datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configurar el MLPClassifier
    mlp = MLPClassifier(max_iter=1000, random_state=1)
    # mlp = MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, random_state=42)

    # Entrenar el modelo en el conjunto de entrenamiento
    mlp.fit(X_train, y_train)
    # print(f"numero de neuronas por capas: {[ i.shape for i in mlp.coefs_ ]}")
    print(f"numero de saldias: {mlp.classes_}")

    # print(mlp.predict(X[0]))

    # Evaluar el rendimiento en el conjunto de prueba
    accuracy = mlp.score(X_test, y_test)

    # Mostrar la tasa de acierto
    print(f'Tasa de acierto en el conjunto de prueba: {accuracy:.2f}')


def ejer2(k=5, type_model="mlp", n_neighbors=3, rs=0):
    data = load_digits()
    X, y = data.data, data.target

    model = get_modelo(type_model, n_neighbors, rs)

    # kf = KFold(n_splits=k)
    # porcentaje_acierto = []
    
    # print(kf)
    # for i, (train_index, test_index) in enumerate(kf.split(X)):
    #     X_train, y_train = X[train_index], y[train_index]
    #     X_test, y_test = X[test_index], y[test_index]

    #     # Entrenar el type_model en el conjunto de entrenamiento
    #     mlp.fit(X_train, y_train)

    #     # Evaluar el rendimiento en el conjunto de prueba
    #     accuracy = mlp.score(X_test, y_test)
    #     porcentaje_acierto.append(accuracy)
    #     print(f'Tasa de acierto en el k: {i} el conjunto de prueba: {accuracy:.2f}')
        # Validación cruzada con KFold
    # print(f'Media tasa de acierto: {np.mean(porcentaje_acierto)} desvio: {np.std(porcentaje_acierto)}')
    
    kfold = KFold(n_splits=k, shuffle=True)
    accuracies = cross_val_score(model, X, y, cv=kfold)
    mean_accuracy = np.mean(accuracies)
    variance_accuracy = np.var(accuracies)
    print('-'*30)
    print(f'[{type_model}]: {k} particiones -> Accuracy: {accuracies}')
    print(f'Mean Accuracy: {mean_accuracy:.2f}, Variance: {variance_accuracy:.4f}')
    

def ejer3(k=5, type_model="mlp", n_neighbors=3, rs=0):
    data = load_wine()
    X, y = data.data, data.target

    # meta_model = BaggingClassifier(
    #     estimator=get_modelo(type_model,n_neighbors,rs),
    #     n_estimators=10,
    #     max_samples=0.32
    # )

    meta_model = AdaBoostClassifier(
        estimator=get_modelo(type_model,n_neighbors,rs,pb=True),
        n_estimators=10
    )

    kfold = KFold(n_splits=k, shuffle=True)
    accuracies = cross_val_score(meta_model, X, y, cv=kfold)
    mean_accuracy = np.mean(accuracies)
    variance_accuracy = np.var(accuracies)
    print('-'*30)
    print(f'[{type_model}]: {k} particiones -> Accuracy: {accuracies}')
    print(f'Mean Accuracy: {mean_accuracy:.2f}, Variance: {variance_accuracy:.4f}')



if __name__ == '__main__':

    # ejer1()

    # ejer2(k=5, type_model="mlp", rs=42)
    # ejer2(k=5, type_model="nb")
    # ejer2(k=5, type_model="lda")
    # ejer2(k=5, type_model="kn")
    # ejer2(k=5, type_model="dt")
    # ejer2(k=5, type_model="svm")

    ejer3(k=5, type_model="nb")
    ejer3(k=5, type_model="dt")
    ejer3(k=5, type_model="svm")
    # k-nearest neighbours no es un estimador base valido para adaboost