import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import logging
from pathlib import Path

class DataLoader:
    """Classe pour charger et prétraiter les données"""
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Charge les données d'entraînement et de test"""
        try:
            X_train = np.load(self.data_dir / 'data_train.npy')
            X_test = np.load(self.data_dir / 'data_test.npy')
            vocab_map = np.load(self.data_dir / 'vocab_map.npy', allow_pickle=True)
            y_train = pd.read_csv(self.data_dir / 'label_train.csv')['label'].values
            
            self.logger.info(f"Données chargées - X_train: {X_train.shape}, X_test: {X_test.shape}")
            return X_train, X_test, y_train, vocab_map
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données: {str(e)}")
            raise

class DataVisualizer:
    """Classe pour la visualisation des données et des résultats"""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        plt.style.use('default')
    
    def plot_class_distribution(self, y: np.ndarray, title: str = "Distribution des Classes"):
        """Visualise la distribution des classes"""
        plt.figure(figsize=(10, 6))
        sns.countplot(x=y)
        plt.title(title)
        plt.xlabel("Classe")
        plt.ylabel("Nombre d'exemples")
        plt.show()
    
    def plot_feature_importance(self, model: Pipeline, vocab_map: np.ndarray, top_n: int = 20):
        """Visualise les features les plus importantes"""
        selector = model.named_steps['selector']
        feature_scores = selector.scores_
        top_indices = np.argsort(feature_scores)[-top_n:]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), feature_scores[top_indices][::-1])
        plt.yticks(range(top_n), [vocab_map[i] for i in top_indices][::-1])
        plt.title(f"Top {top_n} Features les Plus Importantes")
        plt.xlabel("Score Chi2")
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Visualise la matrice de confusion"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion')
        plt.ylabel('Vraie Classe')
        plt.xlabel('Classe Prédite')
        plt.show()

class OptimizedSVMClassifier:
    """Classe principale pour la classification avec SVM optimisé"""
    def __init__(self, n_features: int = 2000):
        self.n_features = n_features
        self.pipeline = None
        self.best_params = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def create_pipeline(self) -> Pipeline:
        """Crée le pipeline optimisé"""
        return Pipeline([
            ('tfidf', TfidfTransformer(sublinear_tf=True)),
            ('selector', SelectKBest(chi2, k=self.n_features)),
            ('scaler', StandardScaler(with_mean=False)),
            ('svm', LinearSVC(
                random_state=42,
                max_iter=1000,
                dual=False,
                class_weight='balanced'
            ))
        ])
    
    def get_param_grid(self) -> Dict[str, Any]:
        """Définit la grille de paramètres pour GridSearchCV"""
        return {
            'tfidf__norm': ['l1', 'l2'],
            'svm__C': [0.1, 1.0, 10.0],
            'selector__k': [1000, 2000]
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Entraîne le modèle avec validation"""
        self.pipeline = self.create_pipeline()
        param_grid = self.get_param_grid()
        
        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            scoring='f1_macro',
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        self.logger.info("Début de l'entraînement...")
        grid_search.fit(X_train, y_train)
        
        self.pipeline = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        metrics = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            metrics['val_f1'] = f1_score(y_val, val_pred, average='macro')
            
        self.logger.info(f"Meilleurs paramètres: {self.best_params}")
        self.logger.info(f"Meilleur score F1-macro: {metrics['best_score']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait des prédictions sur de nouvelles données"""
        if self.pipeline is None:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        return self.pipeline.predict(X)

def main():
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Chargement des données
        data_loader = DataLoader()
        X_train, X_test, y_train, vocab_map = data_loader.load_data()
        
        # Création des visualisations
        visualizer = DataVisualizer()
        visualizer.plot_class_distribution(y_train, "Distribution des Classes d'Entraînement")
        
        # Division train/validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        # Entraînement du modèle
        classifier = OptimizedSVMClassifier(n_features=2000)
        metrics = classifier.train(X_train_final, y_train_final, X_val, y_val)
        
        # Visualisations post-entraînement
        visualizer.plot_feature_importance(classifier.pipeline, vocab_map)
        visualizer.plot_confusion_matrix(y_val, classifier.predict(X_val))
        
        # Prédictions finales
        y_pred = classifier.predict(X_test)
        submission = pd.DataFrame({
            'id': range(len(y_pred)),
            'label': y_pred
        })
        submission.to_csv('submission.csv', index=False)
        logger.info("Fichier de soumission créé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur dans l'exécution principale: {str(e)}")
        raise

if __name__ == "__main__":
    main()
