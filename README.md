# ml_projects

## Table des matières
1. [Introduction](#introduction)
2. [Structure du projet](#structure-du-projet)
   - [Devoir 2 Théorique](#devoir-2-théorique)
   - [Devoir 2 Pratique](#devoir-2-pratique)
   - [Compétition Kaggle](#compétition-kaggle)
3. [Installation](#installation)
4. [Utilisation](#utilisation)
5. [Lien Overleaf](#lien-overleaf)
6. [Lien Kaggle](#lien-kaggle)
7. [Contact](#contact)

## Introduction
Ce dépôt contient plusieurs projets réalisés pour le cours IFT-3395. Il regroupe des travaux théoriques, pratiques ainsi qu'une participation à une compétition Kaggle. L'accent est mis sur la construction d'algorithmes à partir de zéro en utilisant NumPy et Python, sans bibliothèques d'apprentissage automatique.

## Structure du projet

### Devoir 2 Théorique
Ce dossier contient l'analyse théorique du Devoir 2, avec tous les détails dans le document Overleaf.
- [Lien vers le rapport théorique sur Overleaf](https://www.overleaf.com/2667619391xcqccxpbnwcs#d6fc5c)

### Devoir 2 Pratique
Dans ce dossier, vous trouverez la solution au Devoir 2 pratique, qui inclut :
- **`solution.py`** : Le fichier contenant l'implémentation complète de la solution.
- **`rapport_pratique.pdf`** : Le rapport décrivant l'approche, la méthodologie et les résultats.

Dans le cadre de l'entraînement du SVM un-contre-tous avec pénalité L2, Les principales sections abordées incluent :
- Calcul du gradient pour le hinge loss.
- Implémentation des méthodes de régularisation.
- Analyse des graphiques d'évolution de la perte et de l'exactitude.

### Compétition Kaggle
La section Kaggle contient le code nécessaire pour entraîner un modèle de régression logistique sur un ensemble de données réel. Ce projet inclut :
- **`test.py`** : Le script principal pour charger les données, entraîner le modèle, et faire les prédictions finales.
- Le modèle de régression logistique est entraîné à partir de zéro, avec des fonctionnalités telles que la régularisation L1/L2 (Elastic Net) et une recherche d'hyperparamètres.

## Installation
Pour exécuter le projet, suivez les étapes ci-dessous :

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/shollet/ml_projects.git
   ```

2. Installez l'environnement conda :
   ```bash
   conda create -n ift6758-conda-env python=3.8
   conda activate ift6758-conda-env
   ```

3. Installez les dépendances :
   ```bash
   pip install numpy pandas matplotlib
   ```

4. Organisez les données dans le dossier approprié :
   - Placez les fichiers d'entraînement, de validation, et de test dans le dossier `Data_classification/`.

## Utilisation

### Exécution du projet Kaggle :
1. Pour entraîner le modèle et générer des prédictions :
   ```bash
   python test.py
   ```

2. Le script exécutera les étapes suivantes :
   - Charger et prétraiter les données (calcul de TF-IDF, suppression des mots vides, etc.).
   - Entraîner la régression logistique et ajuster les hyperparamètres via une recherche par grille.
   - Sauvegarder les prédictions dans `predictions.csv`.

### Exécution du projet pour le Devoir 2 Pratique :
1. Pour exécuter l'entraînement du SVM un-contre-tous, lancez le script `solution.py` :
   ```bash
   python solution.py
   ```
2. Les résultats incluent des graphiques pour :
   - **Perte d'entraînement** et **perte de validation**.
   - **Exactitude d'entraînement** et **exactitude de validation**.
3. Ces graphiques sont stockés dans le dossier spécifié par le script, et les résultats sont détaillés dans le rapport pratique en PDF.

### Détails des fichiers :
- **`test.py`** : Entraîne le modèle et fait les prédictions pour la compétition Kaggle.
- **`solution.py`** : Implémentation pratique pour le devoir 2, incluant la création des graphiques d'entraînement et de validation.
- **`rapport_pratique.pdf`** : Rapport pour la partie pratique du devoir.
- **`predictions.csv`** : Résultats des prédictions sur l'ensemble de test Kaggle.

## Lien Overleaf
Le rapport théorique est disponible via le lien suivant :
- [Lien Overleaf](https://www.overleaf.com/2667619391xcqccxpbnwcs#d6fc5c)

## Lien Kaggle
Détails de la compétition Kaggle à venir.

## Contact
Pour toute question, vous pouvez me contacter via :
- Email : [shayan.nicolas.hollet@umontreal.ca](mailto:shayan.nicolas.hollet@umontreal.ca)
- GitHub : [shollet](https://github.com/shollet)
