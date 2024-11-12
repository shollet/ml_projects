import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

class DataLoader:
    """
    Classe pour charger, prétraiter et préparer les données de texte pour la classification.
    Cette classe fournit des méthodes pour charger les fichiers de données, supprimer les mots vides,
    appliquer TF-IDF, sélectionner les caractéristiques, normaliser les données, et diviser les ensembles d'entraînement et de validation.
    """

    def __init__(self, data_folder):
        """
        Initialise le DataLoader en définissant le dossier de données et les structures nécessaires.

        Args:
            data_folder (str): Chemin du dossier contenant les fichiers de données.
        """
        self.data_folder = data_folder
        # Ensemble de mots vides courants à supprimer
        self.stop_words = set([...])  # Liste des mots vides
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.vocab_map = None
        self.index_to_word = None
        self.class_weights = None
        self.IDF = None
        self.mean = None
        self.std = None
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        self.train_indices = None
        self.val_indices = None
        self.X_train_split = None
        self.y_train_split = None
        self.X_val_split = None
        self.y_val_split = None

    def load_data(self):
        """
        Charge les fichiers de données, y compris les matrices d'entraînement et de test, les étiquettes, et le vocabulaire.
        """
        # Définition des chemins vers les fichiers
        data_train_path = os.path.join(self.data_folder, 'data_train.npy')
        data_test_path = os.path.join(self.data_folder, 'data_test.npy')
        labels_train_npy_path = os.path.join(self.data_folder, 'labels_train.npy')
        labels_train_csv_path = os.path.join(self.data_folder, 'label_train.csv')
        vocab_map_path = os.path.join(self.data_folder, 'vocab_map.npy')

        # Chargement des matrices de caractéristiques
        self.X_train = np.load(data_train_path, allow_pickle=True)
        self.X_test = np.load(data_test_path, allow_pickle=True)

        # Chargement des étiquettes d'entraînement
        if os.path.exists(labels_train_npy_path):
            self.y_train = np.load(labels_train_npy_path, allow_pickle=True).astype(int)
        elif os.path.exists(labels_train_csv_path):
            labels_df = pd.read_csv(labels_train_csv_path)
            # Vérifie la colonne contenant les étiquettes et l'extrait
            if 'label' in labels_df.columns:
                self.y_train = labels_df['label'].values.astype(int)
            elif 'Label' in labels_df.columns:
                self.y_train = labels_df['Label'].values.astype(int)
            else:
                self.y_train = labels_df.iloc[:, 1].values.astype(int)
        else:
            raise FileNotFoundError("Labels file not found in the data directory.")

        # Chargement de la carte de vocabulaire
        self.vocab_map = self.load_vocab_map(vocab_map_path)
        # Création de la correspondance index-vers-mot
        self.index_to_word = self.create_index_to_word_mapping(self.vocab_map)

        print("Data loading complete.")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"Unique labels in y_train: {np.unique(self.y_train)}")

    def load_vocab_map(self, vocab_map_path):
        """
        Charge la carte de vocabulaire depuis le fichier spécifié.

        Args:
            vocab_map_path (str): Chemin du fichier contenant la carte de vocabulaire.

        Returns:
            dict: Dictionnaire de vocabulaire, chaque mot étant associé à un index unique.
        """
        vocab_data = np.load(vocab_map_path, allow_pickle=True)

        # Différents formats possibles pour vocab_data
        if isinstance(vocab_data, dict):
            vocab_map = vocab_data
        elif isinstance(vocab_data, np.ndarray):
            if vocab_data.dtype == object:
                if isinstance(vocab_data[0], (tuple, list)):
                    vocab_map = dict(vocab_data)
                elif isinstance(vocab_data[0], str):
                    vocab_map = {idx: word for idx, word in enumerate(vocab_data)}
                else:
                    raise ValueError("Unhandled data type in vocab_data[0]")
            else:
                raise ValueError("Unhandled data type in vocab_data")
        else:
            raise ValueError("Unexpected data format in vocab_map.npy")

        return vocab_map

    def create_index_to_word_mapping(self, vocab_map):
        """
        Crée une correspondance index-vers-mot à partir de la carte de vocabulaire.

        Args:
            vocab_map (dict): Dictionnaire de vocabulaire.

        Returns:
            dict: Dictionnaire de correspondance index-vers-mot.
        """
        if isinstance(next(iter(vocab_map.keys())), int):
            return vocab_map
        return {index: word for word, index in vocab_map.items()}

    def preprocess_data(self):
        """
        Effectue toutes les étapes de prétraitement : suppression de mots vides, TF-IDF, sélection de caractéristiques,
        standardisation et ajout d'un biais.
        """
        self.remove_stop_words()
        self.compute_tfidf()
        self.feature_selection()
        self.standardize_features()
        self.add_bias_term()
        self.compute_class_weights()

    def remove_stop_words(self):
        """
        Supprime les mots vides des matrices d'entraînement et de test.
        """
        stop_word_indices = [index for index, word in self.index_to_word.items() if word.lower() in self.stop_words]
        keep_indices = [i for i in range(self.X_train.shape[1]) if i not in stop_word_indices]

        self.X_train = self.X_train[:, keep_indices]
        self.X_test = self.X_test[:, keep_indices]
        self.index_to_word = {new_idx: self.index_to_word[old_idx] for new_idx, old_idx in enumerate(keep_indices)}

        print(f"Stop words removed. Remaining features: {self.X_train.shape[1]}")

    def compute_tfidf(self):
        """
        Calcule la représentation TF-IDF pour les données d'entraînement et de test.
        """
        N = self.X_train.shape[0]
        DF = np.sum(self.X_train > 0, axis=0)
        self.IDF = np.log((N + 1) / (DF + 1)) + 1  # Ajouter 1 pour éviter la division par zéro
        self.X_train_tfidf = self.X_train * self.IDF
        self.X_test_tfidf = self.X_test * self.IDF

        print("TF-IDF computation complete.")

    def feature_selection(self):
        """
        Sélectionne les caractéristiques en fonction des seuils de fréquence de document.
        """
        N = self.X_train.shape[0]
        DF = np.sum(self.X_train > 0, axis=0)
        DF_threshold_low = 5
        DF_threshold_high = N * 0.9
        selected_features = np.where((DF > DF_threshold_low) & (DF < DF_threshold_high))[0]

        self.X_train_tfidf = self.X_train_tfidf[:, selected_features]
        self.X_test_tfidf = self.X_test_tfidf[:, selected_features]
        self.IDF = self.IDF[selected_features]

        print(f"Feature selection complete. Selected features: {self.X_train_tfidf.shape[1]}")

    def standardize_features(self):
        """
        Standardise les caractéristiques en utilisant une normalisation z-score.
        """
        self.mean = np.mean(self.X_train_tfidf, axis=0)
        self.std = np.std(self.X_train_tfidf, axis=0) + 1e-8

        self.X_train_tfidf = (self.X_train_tfidf - self.mean) / self.std
        self.X_test_tfidf = (self.X_test_tfidf - self.mean) / self.std

        print("Feature standardization complete.")

    def add_bias_term(self):
        """
        Ajoute un terme de biais à la matrice de caractéristiques.
        """
        self.X_train_tfidf = np.hstack((np.ones((self.X_train_tfidf.shape[0], 1)), self.X_train_tfidf))
        self.X_test_tfidf = np.hstack((np.ones((self.X_test_tfidf.shape[0], 1)), self.X_test_tfidf))

        print("Bias term added to feature matrices.")

    def compute_class_weights(self):
        """
        Calcule les poids des classes pour compenser le déséquilibre de classe.
        """
        class_counts = np.bincount(self.y_train)
        total_samples = len(self.y_train)
        self.class_weights = {0: total_samples / class_counts[0], 1: total_samples / class_counts[1]}

        print(f"Class weights computed: {self.class_weights}")

    def split_data(self, validation_size=0.2, shuffle=True):
        """
        Divise les données en ensembles d'entraînement et de validation.

        Args:
            validation_size (float): Pourcentage des données à utiliser pour la validation.
            shuffle (bool): Indique si les données doivent être mélangées avant la division.
        """
        m = len(self.y_train)
        indices = np.arange(m)
        if shuffle:
            np.random.shuffle(indices)
        split_point = int(m * (1 - validation_size))
        self.train_indices = indices[:split_point]
        self.val_indices = indices[split_point:]
        self.X_train_split = self.X_train_tfidf[self.train_indices]
        self.y_train_split = self.y_train[self.train_indices]
        self.X_val_split = self.X_train_tfidf[self.val_indices]
        self.y_val_split = self.y_train[self.val_indices]

        print(f"Data split into training ({len(self.train_indices)}) and validation ({len(self.val_indices)}) sets.")

    def get_train_validation_data(self):
        """
        Retourne les données d'entraînement et de validation préparées.

        Returns:
            Tuple[np.ndarray]: Données d'entraînement et de validation (caractéristiques et étiquettes).
        """
        return self.X_train_split, self.y_train_split, self.X_val_split, self.y_val_split

    def get_test_data(self):
        """
        Retourne les données de test préparées.

        Returns:
            np.ndarray: Matrice de caractéristiques des données de test.
        """
        return self.X_test_tfidf


class LogisticRegressionModel:
    """
    Implémentation d'un modèle de régression logistique avec régularisation L2 et optimisation Adam.
    Ce modèle permet d'entraîner sur un ensemble d'entraînement, de rechercher le meilleur seuil pour la classification
    et de faire des prédictions sur de nouvelles données.
    """

    def __init__(self, class_weights, learning_rate=0.0005, num_iterations=5000, lambda_=0.005,
                 batch_size=128, early_stopping_rounds=20):
        """
        Initialise le modèle de régression logistique.

        Args:
            class_weights (dict): Poids des classes pour gérer le déséquilibre.
            learning_rate (float): Taux d'apprentissage pour l'optimisation.
            num_iterations (int): Nombre maximum d'itérations.
            lambda_ (float): Paramètre de régularisation L2.
            batch_size (int): Taille des lots pour la descente de gradient mini-lots.
            early_stopping_rounds (int): Nombre de tours sans amélioration pour arrêter l'entraînement.
        """
        self.class_weights = class_weights
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.theta = None
        self.best_threshold = 0.5
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    @staticmethod
    def sigmoid(z):
        """
        Calcule la fonction sigmoïde pour une entrée donnée.

        Args:
            z (np.ndarray): Entrée pour le calcul de la sigmoïde.

        Returns:
            np.ndarray: Valeurs de la sigmoïde appliquée élément par élément.
        """
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        """
        Calcule la fonction de coût avec régularisation pour la régression logistique.

        Args:
            X (np.ndarray): Matrice de caractéristiques.
            y (np.ndarray): Étiquettes cibles.

        Returns:
            float: Valeur du coût régularisé.
        """
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        epsilon = self.epsilon

        # Calcul du coût pondéré
        cost = -(1 / m) * (
            self.class_weights[1] * np.dot(y, np.log(h + epsilon)) +
            self.class_weights[0] * np.dot((1 - y), np.log(1 - h + epsilon))
        )
        reg_term = (self.lambda_ / (2 * m)) * np.sum(self.theta[1:] ** 2)
        return cost + reg_term

    def compute_gradient(self, X, y):
        """
        Calcule le gradient de la fonction de coût.

        Args:
            X (np.ndarray): Matrice de caractéristiques.
            y (np.ndarray): Étiquettes cibles.

        Returns:
            np.ndarray: Gradient du coût.
        """
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        weights = np.vectorize(self.class_weights.get)(y)
        gradient = (1 / m) * (X.T @ ((h - y) * weights))
        gradient[1:] += (self.lambda_ / m) * self.theta[1:]
        return gradient

    def train(self, X_train, y_train, X_val, y_val):
        """
        Entraîne le modèle de régression logistique en utilisant l'optimiseur Adam.

        Args:
            X_train (np.ndarray): Caractéristiques d'entraînement.
            y_train (np.ndarray): Étiquettes d'entraînement.
            X_val (np.ndarray): Caractéristiques de validation.
            y_val (np.ndarray): Étiquettes de validation.
        """
        m, n = X_train.shape
        self.theta = np.zeros(n)
        m_t = np.zeros(n)
        v_t = np.zeros(n)

        cost_history = []
        val_cost_history = []
        best_val_cost = float('inf')
        best_theta = None
        no_improvement_counter = 0

        print("Starting training...")

        for i in range(1, self.num_iterations + 1):
            permutation = np.random.permutation(m)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for j in range(0, m, self.batch_size):
                X_batch = X_shuffled[j:j + self.batch_size]
                y_batch = y_shuffled[j:j + self.batch_size]

                gradient = self.compute_gradient(X_batch, y_batch)

                # Met à jour les moments biaisés pour Adam
                m_t = self.beta1 * m_t + (1 - self.beta1) * gradient
                v_t = self.beta2 * v_t + (1 - self.beta2) * (gradient ** 2)

                # Correction du biais
                m_hat = m_t / (1 - self.beta1 ** i)
                v_hat = v_t / (1 - self.beta2 ** i)

                # Mise à jour des paramètres
                self.theta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Calcul du coût d'entraînement et de validation
            cost = self.compute_cost(X_train, y_train)
            val_cost = self.compute_cost(X_val, y_val)
            cost_history.append(cost)
            val_cost_history.append(val_cost)

            # Arrêt précoce si aucune amélioration
            if val_cost < best_val_cost - 1e-5:
                best_val_cost = val_cost
                best_theta = self.theta.copy()
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= self.early_stopping_rounds:
                print(f"Early stopping at iteration {i}")
                self.theta = best_theta
                break

            if i % 500 == 0 or i == self.num_iterations:
                print(f"Iteration {i}: Training cost = {cost}, Validation cost = {val_cost}")

        print("Training complete.")

    def find_best_threshold(self, X_val, y_val):
        """
        Trouve le meilleur seuil de classification basé sur le F1 score sur l'ensemble de validation.

        Args:
            X_val (np.ndarray): Caractéristiques de validation.
            y_val (np.ndarray): Étiquettes de validation.
        """
        probabilities = self.sigmoid(X_val @ self.theta)
        thresholds = np.linspace(0.1, 0.9, 81)
        f1_scores = []

        for threshold in thresholds:
            y_pred = (probabilities >= threshold).astype(int)
            f1 = self.f1_score(y_val, y_pred)
            f1_scores.append(f1)

        best_idx = np.argmax(f1_scores)
        self.best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        print(f"Best threshold found: {self.best_threshold}")
        print(f"Best F1 Score on validation set: {best_f1}")

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Calcule le F1 score manuellement.

        Args:
            y_true (np.ndarray): Étiquettes réelles.
            y_pred (np.ndarray): Prédictions.

        Returns:
            float: F1 score.
        """
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1

    def evaluate(self, X, y, data_split='Validation'):
        """
        Évalue le modèle en utilisant le F1 score sur les données spécifiées.

        Args:
            X (np.ndarray): Caractéristiques des données.
            y (np.ndarray): Étiquettes des données.
            data_split (str): Nom du jeu de données (ex: 'Validation').

        Returns:
            float: F1 score.
        """
        probabilities = self.sigmoid(X @ self.theta)
        y_pred = (probabilities >= self.best_threshold).astype(int)
        f1 = self.f1_score(y, y_pred)
        print(f"F1 Score on {data_split} set: {f1}")
        return f1

    def predict(self, X):
        """
        Effectue des prédictions sur de nouvelles données.

        Args:
            X (np.ndarray): Caractéristiques des nouvelles données.

        Returns:
            np.ndarray: Prédictions binaires (0 ou 1).
        """
        probabilities = self.sigmoid(X @ self.theta)
        return (probabilities >= self.best_threshold).astype(int)


if __name__ == "__main__":
    # Initialisation du DataLoader
    data_loader = DataLoader(data_folder='classer-le-text')

    # Chargement et prétraitement des données
    data_loader.load_data()
    data_loader.preprocess_data()
    data_loader.split_data(validation_size=0.2)

    # Récupération des ensembles d'entraînement et de validation
    X_train, y_train, X_val, y_val = data_loader.get_train_validation_data()
    X_test = data_loader.get_test_data()

    # Initialisation et entraînement du modèle
    model = LogisticRegressionModel(class_weights=data_loader.class_weights)
    model.train(X_train, y_train, X_val, y_val)

    # Recherche du meilleur seuil
    model.find_best_threshold(X_val, y_val)

    # Évaluation du modèle
    model.evaluate(X_val, y_val, data_split='Validation')
    model.evaluate(X_train, y_train, data_split='Training')

    # Prédictions sur l'ensemble de test
    y_test_pred = model.predict(X_test)

    # Sauvegarde des prédictions
    test_ids = np.arange(X_test.shape[0])
    predictions_df = pd.DataFrame({
        'ID': test_ids,
        'label': y_test_pred
    })
    predictions_df.to_csv('predictions.csv', index=False)

    print("Predictions saved to 'predictions.csv'.")
