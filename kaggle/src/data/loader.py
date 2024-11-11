from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Optional

load_dotenv()

class DataLoader:
    """Classe pour charger et prétraiter les données"""
    
    def __init__(self, data_dir: str = os.getenv('DATA_DIR', 'data')):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Charge les données d'entraînement et de test"""
        
        # Initialisation des variables à None pour éviter les erreurs de référence
        X_train, X_test, vocab_map, y_train = None, None, None, None
        
        try:
            # Chargement de X_train
            X_train_path = self.data_dir / 'data_train.npy'
            if X_train_path.exists():
                X_train = np.load(X_train_path)
                self.logger.info(f"X_train chargé avec succès, dimensions: {X_train.shape}")
            else:
                self.logger.error(f"Fichier non trouvé: {X_train_path}")
                raise FileNotFoundError(f"Fichier non trouvé: {X_train_path}")
            
            # Chargement de X_test
            X_test_path = self.data_dir / 'data_test.npy'
            if X_test_path.exists():
                X_test = np.load(X_test_path)
                self.logger.info(f"X_test chargé avec succès, dimensions: {X_test.shape}")
            else:
                self.logger.error(f"Fichier non trouvé: {X_test_path}")
                raise FileNotFoundError(f"Fichier non trouvé: {X_test_path}")
            
            # Chargement de vocab_map
            vocab_map_path = self.data_dir / 'vocab_map.npy'
            if vocab_map_path.exists():
                vocab_map = np.load(vocab_map_path, allow_pickle=True)
                self.logger.info(f"vocab_map chargé avec succès, nombre d'entrées: {len(vocab_map)}")
            else:
                self.logger.warning(f"Fichier vocab_map non trouvé: {vocab_map_path}")

            # Chargement de y_train
            y_train_path = self.data_dir / 'label_train.csv'
            if y_train_path.exists():
                y_train = pd.read_csv(y_train_path)['label'].values
                self.logger.info(f"y_train chargé avec succès, dimensions: {y_train.shape}")
            else:
                self.logger.error(f"Fichier non trouvé: {y_train_path}")
                raise FileNotFoundError(f"Fichier non trouvé: {y_train_path}")

            return X_train, X_test, y_train, vocab_map
        
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données: {str(e)}")
            raise
