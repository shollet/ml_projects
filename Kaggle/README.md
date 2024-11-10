# Projet de Classification de Texte - Compétition Kaggle IFT-3395/6390

Ce projet implémente un système de classification de texte basé sur SVM pour la compétition Kaggle du cours IFT-3395/6390. L'objectif est de trier automatiquement des documents courts en utilisant des vecteurs de comptage de termes.

## Structure du Projet

```
project/
├── data/                # Données de la compétition
│   ├── data_test.npy   # Données de test
│   ├── data_train.npy  # Données d'entraînement 
│   ├── label_train.csv # Étiquettes d'entraînement
│   └── vocab_map.npy   # Mapping du vocabulaire
│
├── src/                # Code source
│   ├── data/          # Chargement des données
│   ├── models/        # Modèles de classification (SVM)
│   ├── utils/         # Logging et utilitaires
│   └── visualization/ # Visualisations
│
├── logs/              # Fichiers de log d'entraînement
├── output/            # Résultats et visualisations
└── submissions/       # Fichiers de soumission Kaggle
```

## Configuration

### Environnement
1. Créez l'environnement conda :
```bash
conda create -n ift6758-conda-env python=3.8
conda activate ift6758-conda-env
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Entraînement et Prédictions
```bash
python train.py
```

Le script exécute séquentiellement :
1. Chargement et prétraitement des données vectorielles
2. Entraînement d'un SVM optimisé avec :
   - Transformation TF-IDF
   - Sélection de features (Chi2)
   - Standardisation
   - Classification linéaire
3. Génération des visualisations
4. Création du fichier de soumission Kaggle

### Sorties
- `output/` : Visualisations et analyses
  - Distribution des classes
  - Importance des features
  - Matrices de confusion
- `logs/` : Logs détaillés d'entraînement
- `submissions/` : Fichiers CSV pour Kaggle

## Caractéristiques Principales

### Prétraitement
- Vectorisation TF-IDF des données textuelles
- Sélection des features les plus pertinentes
- Standardisation des données

### Modélisation
- SVM linéaire optimisé
- Validation croisée stratifiée
- Gestion du déséquilibre des classes
- Recherche d'hyperparamètres automatisée

### Monitoring
- Logging détaillé de l'entraînement
- Visualisations des performances
- Suivi des métriques (F1-score macro)

## Contact
Pour toute question :
- Email : [shayan.nicolas.hollet@umontreal.ca](mailto:shayan.nicolas.hollet@umontreal.ca)
- GitHub : [shollet](https://github.com/shollet)
