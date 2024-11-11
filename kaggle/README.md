# Projet de Classification de Texte - Compétition Kaggle IFT-3395/6390

Ce projet implémente un système de classification de texte basé sur des modèles SVM et Régression Logistique pour la compétition Kaggle du cours IFT-3395/6390. L'objectif est de trier automatiquement des documents courts en utilisant des vecteurs de comptage de termes.

## Structure du Projet

```
project/
├── data/                # Données de la compétition
│   ├── data_test.npy    # Données de test
│   ├── data_train.npy   # Données d'entraînement 
│   ├── label_train.csv  # Étiquettes d'entraînement
│   └── vocab_map.npy    # Mapping du vocabulaire
│
├── src/                 # Code source
│   ├── data/            # Chargement des données
│   ├── models/          # Modèles de classification
│   │   ├── svm.py       # Modèle SVM optimisé
│   │   └── logistic.py  # Modèle de Régression Logistique optimisé
│   ├── utils/           # Logging et utilitaires
│   └── visualization/   # Visualisations
│
├── logs/                # Fichiers de log d'entraînement (dernier log : `latest.log`)
├── output/              # Résultats et visualisations
│   ├── class_distribution.png
│   ├── feature_importance.png
│   └── submission.csv   # Fichier de soumission Kaggle
├── docker-compose.yml   # Configuration de Docker Compose
└── .env                 # Variables d'environnement
```

## Configuration

### Variables d'Environnement

Les configurations d'hyperparamètres et chemins importants sont gérés par un fichier `.env`. Créez un fichier `.env` à la racine du projet avec les variables suivantes pour ajuster facilement les paramètres d'entraînement :

```dotenv
# Chemins des répertoires
DATA_DIR=./data
OUTPUT_DIR=./output
LOGS_DIR=./logs

# Limites Docker
DOCKER_MEMORY=8g
DOCKER_CPUS=1.0

# Hyperparamètres pour SVM
N_FEATURES_SVM=500
MAX_ITER_SVM=2000
SVM_C_VALUES=0.1,1.0

# Hyperparamètres pour Régression Logistique
N_FEATURES_LOGISTIC=200
MAX_ITER_LOGISTIC=1000
LOGISTIC_C_VALUES=0.01,0.1,1.0
```

### Environnement Conda

1. Créez et activez un environnement conda :
   ```bash
   conda create -n kaggle_text_classification python=3.12
   conda activate kaggle_text_classification
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

### Docker

Pour une exécution Docker, assurez-vous que Docker et Docker Compose sont installés. Utilisez la commande suivante pour construire et exécuter le projet :

```bash
docker compose up --build
```

## Utilisation

### Entraînement et Prédictions
```bash
python train.py
```

Le script exécute les étapes suivantes :
1. Chargement et prétraitement des données textuelles.
2. Entraînement de deux modèles de classification optimisés :
   - **SVM** avec transformation TF-IDF, sélection de features (Chi2), standardisation et classification linéaire.
   - **Régression Logistique** avec des étapes de prétraitement similaires.
3. Évaluation des modèles sur des métriques de performance (F1-score macro).
4. Visualisations et génération du fichier de soumission pour Kaggle.

### Sorties
- **`output/`** : Contient les visualisations et le fichier `submission.csv`.
  - `class_distribution.png` : Distribution des classes dans l'ensemble d'entraînement.
  - `feature_importance.png` : Importance des features sélectionnées.
- **`logs/`** : Fichier `latest.log` qui enregistre les détails de l'exécution pour suivi.
- **`submission.csv`** : Fichier pour soumission à Kaggle.

## Caractéristiques Principales

### Prétraitement des Données
- **TF-IDF** : Transformation de fréquence de mots avec parcimonie de mémoire.
- **Sélection de Features** : Sélection de `n_features` optimales via Chi2.
- **Standardisation** : Standardisation des valeurs pour améliorer la convergence des modèles.

### Modélisation et Optimisation
- **SVM Linéaire et Régression Logistique** : Deux modèles pour comparaison des performances.
- **Recherche d'Hyperparamètres** : `GridSearchCV` pour la recherche d'hyperparamètres, avec nombre de caractéristiques et valeur de régularisation.
- **Évaluation des Modèles** : Validation croisée stratifiée et comparaison des scores F1 pour les deux modèles.

### Suivi et Visualisation
- **Logging** : Un fichier `latest.log` qui enregistre chaque exécution avec les étapes et les erreurs éventuelles.
- **Visualisations** : Graphiques d'importance des features et distribution des classes pour compréhension approfondie des données.

## Contact
Pour toute question, vous pouvez me contacter via :
- Email : [shayan.nicolas.hollet@umontreal.ca](mailto:shayan.nicolas.hollet@umontreal.ca)
- GitHub : [shollet](https://github.com/shollet)
