import logging
from dotenv import load_dotenv
import os
from src.data.loader import DataLoader
from src.models.svm import OptimizedSVMClassifier
from src.models.logistic import OptimizedLogisticClassifier
from src.visualization.visualizer import DataVisualizer
from src.utils.logger import setup_logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
from pathlib import Path

# Charger les variables d'environnement
load_dotenv()

# Récupérer les paramètres du nombre de caractéristiques pour chaque modèle
n_features_svm = int(os.getenv("N_FEATURES_SVM", 200))
n_features_logistic = int(os.getenv("N_FEATURES_LOGISTIC", 200))

def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name="Model"):
    """Entraîne et évalue un modèle donné, et retourne les métriques."""
    logger = logging.getLogger(model_name)
    logger.info(f"Début de l'entraînement pour {model_name}...")
    metrics = model.train(X_train, y_train, X_val, y_val)
    
    # Calcul et affichage du score de validation
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, y_pred, average='macro')
        metrics['val_f1'] = val_f1
        logger.info(f"{model_name} - Score F1 sur la validation: {val_f1:.4f}")
    
    return metrics

def main():
    # Configuration du logging
    logger = setup_logging()
    logger.info("Début de l'exécution")

    # Création des répertoires nécessaires
    output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
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
        
        # Modèle SVM
        svm_model = OptimizedSVMClassifier(n_features=n_features_svm)
        svm_metrics = train_and_evaluate(svm_model, X_train_final, y_train_final, X_val, y_val, model_name="SVM")
        
        # Modèle Logistic Regression
        logistic_model = OptimizedLogisticClassifier(n_features=n_features_logistic)
        logistic_metrics = train_and_evaluate(logistic_model, X_train_final, y_train_final, X_val, y_val, model_name="Logistic Regression")
        
        # Comparaison des résultats
        logger.info("Comparaison des modèles :")
        logger.info(f"SVM - Score F1: {svm_metrics.get('val_f1', 'N/A')}, Temps d'entraînement: {svm_metrics['training_time']:.2f}s")
        logger.info(f"Logistic Regression - Score F1: {logistic_metrics.get('val_f1', 'N/A')}, Temps d'entraînement: {logistic_metrics['training_time']:.2f}s")
        
        # Visualisations post-entraînement pour le modèle choisi (par ex., le SVM)
        visualizer.plot_feature_importance(svm_model.pipeline, vocab_map)
        
        # Prédictions finales avec le modèle SVM comme exemple
        y_pred = svm_model.predict(X_test)
        
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
