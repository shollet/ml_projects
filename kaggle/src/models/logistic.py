import os
from dotenv import load_dotenv
import logging
from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import numpy as np
import time
import warnings

load_dotenv()

class OptimizedLogisticClassifier:
    """Classe principale pour la classification avec Régression Logistique optimisée"""
    
    def __init__(self, n_features: int = int(os.getenv("N_FEATURES_LOGISTIC", 200))):
        self.n_features = n_features
        self.pipeline = None
        self.best_params = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_iter = int(os.getenv("MAX_ITER_LOGISTIC", 1000))
        self.c_values = list(map(float, os.getenv("LOGISTIC_C_VALUES", "0.01,0.1,1.0").split(",")))

    def create_pipeline(self) -> Pipeline:
        model = LogisticRegression(
            max_iter=self.max_iter,
            class_weight='balanced',
            random_state=42
        )
        return Pipeline([
            ('tfidf', TfidfTransformer(sublinear_tf=True)),
            ('selector', SelectKBest(chi2, k=self.n_features)),
            ('scaler', StandardScaler(with_mean=False)),
            ('logistic', model)
        ])

    def get_param_grid(self) -> Dict[str, Any]:
        return {'logistic__C': self.c_values}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Entraîne le modèle avec validation"""
        start_time = time.time()
        self.logger.info("Début de l'entraînement du modèle de Régression Logistique...")

        try:
            self.pipeline = self.create_pipeline()
            param_grid = self.get_param_grid()
            
            grid_search = GridSearchCV(
                estimator=self.pipeline,
                param_grid=param_grid,
                scoring='f1_macro',
                cv=2,
                n_jobs=-1,
                verbose=2
            )
            
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            self.best_params = grid_search.best_params_

            metrics = {
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'training_time': time.time() - start_time
            }

            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                metrics['val_f1'] = f1_score(y_val, val_pred, average='macro')
                
            self.logger.info(f"Entraînement terminé en {metrics['training_time']:.2f}s")
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
