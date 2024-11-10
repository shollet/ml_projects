from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import logging
from typing import Dict, Any, Optional
import numpy as np

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
                max_iter=5000,  # Augmenté pour résoudre le problème de convergence
                dual=False,
                class_weight='balanced',
                tol=1e-4        # Ajusté pour améliorer la convergence
            ))
        ])
    
    def get_param_grid(self) -> Dict[str, Any]:
        """Définit la grille de paramètres pour GridSearchCV"""
        return {
            'tfidf__norm': ['l2'],  # Simplifié pour accélérer
            'svm__C': [0.1, 1.0, 10.0],
            'svm__tol': [1e-4, 1e-3]  # Ajout du paramètre de tolérance
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Entraîne le modèle avec validation"""
        start_time = time.time()
        self.logger.info("Début de l'entraînement du modèle SVM...")
        
        try:
            self.pipeline = self.create_pipeline()
            param_grid = self.get_param_grid()
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                grid_search = GridSearchCV(
                    estimator=self.pipeline,
                    param_grid=param_grid,
                    scoring='f1_macro',
                    cv=3,
                    n_jobs=-1,
                    verbose=2
                )
                
                grid_search.fit(X_train, y_train)
                
                if len(w) > 0:
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
                
                # Rapport détaillé
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
