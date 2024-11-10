import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple

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
