from src.data.loader import DataLoader
from src.models.svm import OptimizedSVMClassifier
from src.visualization.visualizer import DataVisualizer
from src.utils.logger import setup_logging
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

def main():
    # Configuration du logging
    logger = setup_logging()
    logger.info("Début de l'exécution")
    
    # Création des répertoires nécessaires
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Chargement des données
        data_loader = DataLoader()
        X_train, X_test, y_train, vocab_map = data_loader.load_data()
        
        # Visualisations initiales
        visualizer = DataVisualizer()
        visualizer.plot_class_distribution(y_train, "Distribution des Classes d'Entraînement")
        
        # Division train/validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        logger.info(f"Données divisées - Train: {X_train_final.shape}, Validation: {X_val.shape}")
        
        # Entraînement du modèle
        classifier = OptimizedSVMClassifier(n_features=2000)
        metrics = classifier.train(X_train_final, y_train_final, X_val, y_val)
        
        # Visualisations post-entraînement
        visualizer.plot_feature_importance(classifier.pipeline, vocab_map)
        
        # Prédictions finales
        logger.info("Génération des prédictions finales...")
        y_pred = classifier.predict(X_test)
        
        # Sauvegarde des résultats
        submission = pd.DataFrame({
            'id': range(len(y_pred)),
            'label': y_pred
        })
        submission.to_csv(output_dir / 'submission.csv', index=False)
        logger.info(f"Soumission sauvegardée dans {output_dir}")
        
        logger.info("Exécution terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur dans l'exécution principale: {str(e)}")
        raise

if __name__ == "__main__":
    main()
