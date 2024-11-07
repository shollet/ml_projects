import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import os

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
        self.X = [' '.join(map(str, row)) for row in self.X]
        self.X_test = [' '.join(map(str, row)) for row in self.X_test]

        # Initialiser le vectoriseur TF-IDF avec suppression des stop words
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # Limite de features pour éviter les surajustements
        self.X = vectorizer.fit_transform(self.X)
        self.X_test = vectorizer.transform(self.X_test)

        print("TF-IDF transformation complete.")
        print(f"Transformed X shape: {self.X.shape}")
        print(f"Transformed X_test shape: {self.X_test.shape}")

class NaiveBayesKFold:
    def __init__(self, n_splits=5):
        self.model = MultinomialNB()
        self.n_splits = n_splits
        self.scores = []

    def cross_validate(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits)
        fold = 1

        for train_index, val_index in skf.split(X, y):
            print(f"Training on fold {fold}...")
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Entraînement
            self.model.fit(X_train, y_train)

            # Prédictions et score
            y_pred = self.model.predict(X_val)
            score = f1_score(y_val, y_pred, average='macro')
            self.scores.append(score)

            print(f"F1 Score for fold {fold}: {score}")
            fold += 1

        # Moyenne des scores sur tous les folds
        mean_score = np.mean(self.scores)
        print(f"\nAverage F1 Score across {self.n_splits} folds: {mean_score}")
        return mean_score

    def predict(self, X):
        """Predict using the fitted model after cross-validation."""
        return self.model.predict(X)

if __name__ == "__main__":
    # Initialize data loader
    data_loader = DataLoader(data_folder='Kaggle')

    # Load and preprocess data
    data_loader.load_data()
    data_loader.preprocess_data()

    # Initialize and run k-fold cross-validation
    model = NaiveBayesKFold(n_splits=10)
    mean_f1_score = model.cross_validate(data_loader.X, data_loader.y)

    # Make predictions on the test set after cross-validation
    y_test_pred = model.predict(data_loader.X_test)

    # Save predictions to CSV
    test_ids = np.arange(data_loader.X_test.shape[0])
    predictions_df = pd.DataFrame({
        'ID': test_ids,
        'label': y_test_pred
    })
    predictions_df.to_csv('predictions.csv', index=False)

    print("Predictions saved to 'predictions.csv'.")
