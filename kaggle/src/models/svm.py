import os
from dotenv import load_dotenv
import logging
import time
import warnings
from typing import Dict, Any, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report

load_dotenv()

class OptimizedSVMClassifier:
    """Classe principale pour la classification avec SVM optimisé"""

    def __init__(self, n_features: int = int(os.getenv("N_FEATURES_SVM", 500))):
        self.n_features = n_features
        self.pipeline: Optional[Pipeline] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_iter = int(os.getenv("MAX_ITER_SVM", 2000))
        self.c_values = list(map(float, os.getenv("SVM_C_VALUES", "0.1,1.0").split(",")))

    def create_pipeline(self) -> Pipeline:
        return Pipeline([
            ('tfidf', TfidfTransformer(sublinear_tf=True)),
            ('selector', SelectKBest(chi2, k=self.n_features)),
            ('scaler', StandardScaler(with_mean=False)),
            ('svm', LinearSVC(
                random_state=42,
                max_iter=self.max_iter,
                dual=False,
                class_weight='balanced',
                tol=1e-3
            ))
        ])

    def get_param_grid(self) -> Dict[str, Any]:
        return {'svm__C': self.c_values}

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Entraîne le modèle avec validation"""
        start_time = time.time()
        self.logger.info("Début de l'entraînement du modèle SVM optimisé pour faible consommation mémoire...")

        try:
            self.pipeline = self.create_pipeline()
            param_grid = self.get_param_grid()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                grid_search = GridSearchCV(
                    estimator=self.pipeline,
                    param_grid=param_grid,
                    scoring='f1_macro',
                    cv=2,
                    n_jobs=-1,
                    verbose=2
                )

                grid_search.fit(X_train, y_train)

                if w:
                    self.logger.warning("Avertissements pendant l'entraînement:")
                    for warning in w:
                        self.logger.warning(str(warning.message))

            self.pipeline = grid_search.best_estimator_
            self.best_params = grid_search.best_params_

            training_time = time.time() - start_time

            metrics = {
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'training_time': training_time
            }

            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                metrics['val_f1'] = f1_score(y_val, val_pred, average='macro')

                report = classification_report(y_val, val_pred)
                self.logger.info(f"\nRapport de classification:\n{report}")

            self.logger.info(f"Entraînement terminé en {training_time:.2f}s")
            self.logger.info(f"Meilleurs paramètres: {self.best_params}")
            self.logger.info(f"Meilleur score F1-macro: {metrics['best_score']:.4f}")

            return metrics

        except Exception as e:
            self.logger.error(f"Erreur pendant l'entraînement: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        return self.pipeline.predict(X)
