import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import os

# Classe pour charger et prétraiter les données
class DataLoader:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.X = None
        self.y = None
        self.X_test = None

    def load_data(self):
        data_train_path = os.path.join(self.data_folder, 'data_train.npy')
        data_test_path = os.path.join(self.data_folder, 'data_test.npy')
        labels_train_csv_path = os.path.join(self.data_folder, 'label_train.csv')

        # Charger les données d'entraînement et de test
        self.X = np.load(data_train_path, allow_pickle=True)
        self.X_test = np.load(data_test_path, allow_pickle=True)

        # Charger les labels
        labels_df = pd.read_csv(labels_train_csv_path)
        self.y = labels_df['label'].values.astype(int)

        print("Data loading complete.")
        print(f"X shape: {self.X.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"y shape: {self.y.shape}")

    def preprocess_data(self):
        # Convertir les données en format texte pour le vecteur TF-IDF
        X_text = [' '.join(map(str, row)) for row in self.X]
        X_test_text = [' '.join(map(str, row)) for row in self.X_test]

        # Initialiser le vectoriseur TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.X = vectorizer.fit_transform(X_text)
        self.X_test = vectorizer.transform(X_test_text)

        # Calculer des caractéristiques supplémentaires pour l'ensemble d'entraînement
        text_lengths_train = np.array(self.X.sum(axis=1)).flatten()  # Somme des poids TF-IDF pour chaque document
        unique_words_train = np.array((self.X > 0).sum(axis=1)).flatten()  # Nombre de mots uniques (non nuls)
        mean_tfidf_train = np.array(self.X.mean(axis=1)).flatten()  # Moyenne des poids TF-IDF pour chaque document

        # Calculer des caractéristiques supplémentaires pour l'ensemble de test
        text_lengths_test = np.array(self.X_test.sum(axis=1)).flatten()
        unique_words_test = np.array((self.X_test > 0).sum(axis=1)).flatten()
        mean_tfidf_test = np.array(self.X_test.mean(axis=1)).flatten()

        # Normalisation L2
        normalizer = Normalizer(norm='l2')
        self.X = normalizer.fit_transform(self.X)
        self.X_test = normalizer.transform(self.X_test)

        # Ajuster dynamiquement n_components pour TruncatedSVD
        n_components = min(100, self.X.shape[1])  # Limiter à la dimension actuelle si < 100
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.X = svd.fit_transform(self.X)
        self.X_test = svd.transform(self.X_test)

        # Combiner les nouvelles caractéristiques avec la matrice réduite
        self.X = np.hstack((self.X, text_lengths_train.reshape(-1, 1), unique_words_train.reshape(-1, 1), mean_tfidf_train.reshape(-1, 1)))
        self.X_test = np.hstack((self.X_test, text_lengths_test.reshape(-1, 1), unique_words_test.reshape(-1, 1), mean_tfidf_test.reshape(-1, 1)))

        print("Preprocessing complete.")
        print(f"Processed X shape: {self.X.shape}")
        print(f"Processed X_test shape: {self.X_test.shape}")

# Fonction pour entraîner et évaluer plusieurs modèles avec validation croisée
def evaluate_models(X, y):
    models = {
        'Naive Bayes': (GaussianNB(), {}),  # Utiliser GaussianNB qui supporte les valeurs négatives
        'SVM': (LinearSVC(class_weight='balanced', max_iter=5000), {'C': [0.1, 1, 10]}),
        'Random Forest': (RandomForestClassifier(class_weight='balanced', random_state=42), {'n_estimators': [50, 100]}),
        'Logistic Regression': (LogisticRegression(class_weight='balanced', max_iter=5000), {'C': [0.1, 1, 10]}),
        'Ridge Classifier': (RidgeClassifier(class_weight='balanced'), {'alpha': [0.1, 1, 10]}),
        'MLP': (MLPClassifier(hidden_layer_sizes=(100,), max_iter=500), {'alpha': [0.0001, 0.001, 0.01]}),
        'k-NN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]})
    }
    
    best_model = None
    best_score = 0
    best_model_name = None

    for model_name, (model, param_grid) in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Utiliser GridSearchCV pour trouver les meilleurs paramètres
        grid_search = GridSearchCV(model, param_grid, scoring='f1_macro', cv=StratifiedKFold(n_splits=5))
        grid_search.fit(X, y)
        
        # Enregistrer les meilleurs résultats
        mean_score = grid_search.best_score_
        print(f"Best F1 Score for {model_name}: {mean_score} with params {grid_search.best_params_}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = grid_search.best_estimator_
            best_model_name = model_name

    print(f"\nBest model: {best_model_name} with F1 Score: {best_score}")
    return best_model, best_model_name

if __name__ == "__main__":
    # Initialiser le chargeur de données
    data_loader = DataLoader(data_folder="Kaggle")

    # Charger et prétraiter les données
    data_loader.load_data()
    data_loader.preprocess_data()

    # Entraîner et évaluer tous les modèles et sélectionner le meilleur
    best_model, best_model_name = evaluate_models(data_loader.X, data_loader.y)

    # Prédictions finales sur l'ensemble de test avec le meilleur modèle
    y_test_pred = best_model.predict(data_loader.X_test)

    # Sauvegarder les prédictions dans un fichier CSV
    predictions_df = pd.DataFrame({
        'ID': np.arange(len(y_test_pred)),
        'label': y_test_pred
    })
    predictions_df.to_csv(f'{best_model_name.lower()}_best_predictions.csv', index=False)

    print(f"Predictions saved to '{best_model_name.lower()}_best_predictions.csv'.")
