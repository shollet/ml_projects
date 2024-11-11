import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
import logging
from pathlib import Path

class DataVisualizer:
    def __init__(self, output_dir: str = "output"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('ggplot') 
    
    def plot_class_distribution(self, y: np.ndarray, title: str = "Distribution des Classes"):
        """Visualise la distribution des classes."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            unique, counts = np.unique(y, return_counts=True)
            ax.bar(unique, counts, color='skyblue')
            ax.set_title(title)
            ax.set_xlabel("Classe")
            ax.set_ylabel("Nombre d'exemples")
            ax.set_xticks(unique)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Sauvegarder le plot
            output_path = self.output_dir / 'class_distribution.png'
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"Distribution des classes sauvegardée dans {output_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la visualisation de la distribution des classes: {str(e)}")
    
    def plot_feature_importance(self, model: Pipeline, vocab_map: np.ndarray, top_n: int = 20):
        """Visualise les features les plus importantes."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            selector = model.named_steps.get('selector')
            if selector is None:
                raise ValueError("Le modèle ne contient pas de sélecteur nommé 'selector'.")
            
            feature_scores = selector.scores_
            top_indices = np.argsort(feature_scores)[-top_n:]
            
            ax.barh(range(top_n), feature_scores[top_indices][::-1], color='salmon')
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([vocab_map[i] for i in top_indices][::-1])
            ax.set_title(f"Top {top_n} Features les Plus Importantes")
            ax.set_xlabel("Score Chi2")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = self.output_dir / 'feature_importance.png'
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"Importance des features sauvegardée dans {output_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la visualisation de l'importance des features: {str(e)}")
