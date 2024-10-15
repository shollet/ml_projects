import numpy as np
import pandas as pd

# 1. Chargement des données

# a. Chargement des données d'entraînement et de test
X_train = np.load('data_train.npy')
X_test = np.load('data_test.npy')

# b. Chargement des étiquettes d'entraînement en gérant l'en-tête
labels_df = pd.read_csv('label_train.csv')  # Laisse pandas détecter l'en-tête automatiquement
if 'Label' in labels_df.columns:
    y_train = labels_df['Label'].values
elif 'label' in labels_df.columns:
    y_train = labels_df['label'].values
else:
    # Si les colonnes n'ont pas de noms, mais qu'il y a une ligne d'en-tête
    y_train = labels_df.iloc[:, 1].values  # Suppose que les étiquettes sont dans la deuxième colonne

# Conversion des étiquettes en entiers
y_train = y_train.astype(int)

# Vérification des types de données
print("Type de données de y_train :", y_train.dtype)

# Vérification des dimensions initiales
print("Dimension initiale de X_train :", X_train.shape)
print("Dimension initiale de y_train :", y_train.shape)
print("Dimension de X_test :", X_test.shape)

# c. Vérification et ajustement des dimensions si nécessaire
min_samples = min(X_train.shape[0], y_train.shape[0])
X_train = X_train[:min_samples, :]
y_train = y_train[:min_samples]

# Vérification des dimensions après ajustement
print("Dimension de X_train après ajustement :", X_train.shape)
print("Dimension de y_train après ajustement :", y_train.shape)

# Vérification des valeurs uniques dans y_train
unique_labels = np.unique(y_train)
print("Valeurs uniques dans y_train :", unique_labels)

# 2. Prétraitement des données

# a. Ajout de la colonne de biais (interception) aux données d'entraînement et de test
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# b. Normalisation des caractéristiques (excluant la colonne de biais)
def standardize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X_standardized = (X - mean) / std
    return X_standardized

X_train[:, 1:] = standardize_features(X_train[:, 1:])
X_test[:, 1:] = standardize_features(X_test[:, 1:])

# c. Sélection de caractéristiques

# Calcul de la variance de chaque caractéristique
variance = np.var(X_train[:, 1:], axis=0)

# Seuil de variance minimale
variance_threshold = 0.01  # À ajuster

# Sélection des caractéristiques avec une variance suffisante
selected_features = np.where(variance > variance_threshold)[0]

# Mise à jour de X_train et X_test
X_train = np.hstack((
    X_train[:, [0]],  # Colonne de biais
    X_train[:, 1:][:, selected_features]
))

X_test = np.hstack((
    X_test[:, [0]],  # Colonne de biais
    X_test[:, 1:][:, selected_features]
))

# 3. Initialisation des paramètres
theta = np.zeros(X_train.shape[1])

# 4. Définition des fonctions nécessaires

# a. Fonction sigmoïde
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# b. Fonction de coût avec régularisation
def compute_cost_reg(X, y, theta, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = -(1/m) * (np.dot(y, np.log(h + epsilon)) + np.dot((1 - y), np.log(1 - h + epsilon)))
    reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost + reg_term

# c. Gradient de la fonction de coût avec régularisation
def compute_gradient_reg(X, y, theta, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    gradient = (1/m) * (X.T @ (h - y))
    gradient[1:] += (lambda_ / m) * theta[1:]
    return gradient

# d. Fonction de descente de gradient avec mini-lots
def mini_batch_gradient_descent(X, y, theta, learning_rate, num_iterations, lambda_, batch_size):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            gradient = compute_gradient_reg(X_batch, y_batch, theta, lambda_)
            theta -= learning_rate * gradient

        cost = compute_cost_reg(X, y, theta, lambda_)
        cost_history.append(cost)

        if i % 100 == 0 or i == num_iterations - 1:
            print(f"Iteration {i}: Coût régularisé = {cost}")

    return theta, cost_history

# e. Fonction de prédiction
def predict(X, theta, threshold=0.5):
    probabilities = sigmoid(X @ theta)
    return (probabilities >= threshold).astype(int)

# f. Fonction pour calculer le F1 score manuellement
def f1_score_manual(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

# 5. Entraînement du modèle

# Hyperparamètres optimisés
learning_rate = 0.01
num_iterations = 2000
lambda_ = 0.01
batch_size = 64

theta, cost_history = mini_batch_gradient_descent(
    X_train, y_train, theta, learning_rate, num_iterations, lambda_, batch_size
)

# 6. Évaluation du modèle sur l'ensemble d'entraînement

y_train_pred = predict(X_train, theta)
f1 = f1_score_manual(y_train, y_train_pred)
print(f"F1 Score sur l'ensemble d'entraînement : {f1}")

# 7. Prédictions sur l'ensemble de test

y_test_pred = predict(X_test, theta)

# 8. Sauvegarde des prédictions pour soumission

# a. Génération de l'ID pour chaque exemple de test
test_ids = np.arange(0, X_test.shape[0])

# b. Création d'un DataFrame avec les colonnes 'ID' et 'label'
predictions_df = pd.DataFrame({
    'ID': test_ids,
    'label': y_test_pred
})

# c. Sauvegarde du DataFrame dans un fichier CSV sans index ni en-tête
predictions_df.to_csv('predictions.csv', index=False)
