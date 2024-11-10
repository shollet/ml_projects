import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
import logging
from pathlib import Path

class DataVisualizer:
    def __init__(self, output_dir: str = "output"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('default')
    
    def plot_class_distribution(self, y: np.ndarray, title: str = "Distribution des Classes"):
        """Visualise la distribution des classes"""
        fig = plt.figure(figsize=(10, 6))
        unique, counts = np.unique(y, return_counts=True)
        plt.bar(unique, counts)
        plt.title(title)
        plt.xlabel("Classe")
        plt.ylabel("Nombre d'exemples")
        plt.xticks(unique)
        plt.grid(True, alpha=0.3)
        
        # Sauvegarder le plot
        plt.savefig(self.output_dir / 'class_distribution.png')
        plt.close(fig)
        self.logger.info(f"Distribution des classes sauvegardée dans {self.output_dir}")
        
    def plot_feature_importance(self, model: Pipeline, vocab_map: np.ndarray, top_n: int = 20):
        """Visualise les features les plus importantes"""
        fig = plt.figure(figsize=(12, 8))
        selector = model.named_steps['selector']
        feature_scores = selector.scores_
        top_indices = np.argsort(feature_scores)[-top_n:]
        
        plt.barh(range(top_n), feature_scores[top_indices][::-1])
        plt.yticks(range(top_n), [vocab_map[i] for i in top_indices][::-1])
        plt.title(f"Top {top_n} Features les Plus Importantes")
        plt.xlabel("Score Chi2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'feature_importance.png')
        plt.close(fig)
        self.logger.info(f"Importance des features sauvegardée dans {self.output_dir}")
