# Projet final du cours INF5182 : Pipeline CNN Complet – Classification d’images

## Description

Ce notebook implémente un pipeline complet de classification d’images :
1. **Partie 1** : Entraînement d’un CNN personnalisé depuis zéro.
2. **Partie 2** : Extraction de caractéristiques avec des architectures pré-entraînées (ResNet, AlexNet, VGGNet, GoogLeNet, MobileNet, LeNet-5) + classifieurs ML (Decision Tree, AdaBoost, XGBoost, LightGBM, KNN).

## Structure des dossiers

Le dossier racine doit contenir un dossier `data/` organisé comme suit :
data/
├── garbage_classification/
│ ├── cardboard/
│ ├── glass/
│ ├── metal/
│ ├── paper/
│ ├── plastic/
│ └── trash/
├── intel_image/
│ ├── buildings/
│ ├── forest/
│ ├── glacier/
│ ├── mountain/
│ ├── sea/
│ └── street/
├── lung_colon_cancer/
│ ├── colon_aca/
│ ├── colon_n/
│ ├── lung_aca/
│ ├── lung_n/
│ └── lung_scc/
└── wildfire/
├── fire/
└── nofire/

Chaque sous-dossier correspond à une classe. Les noms des sous-dossiers sont utilisés comme libellés.

## Installation des dépendances

### Versions requises (testées)

| Bibliothèque   | Version |
|----------------|---------|
| Python         | 3.11+   |
| torch          | 2.0+    |
| torchvision    | 0.15+   |
| scikit-learn   | 1.2+    |
| xgboost        | 1.7+    |
| lightgbm       | 3.3+    |
| pandas         | 1.5+    |
| numpy          | 1.23+   |
| matplotlib     | 3.6+    |
| seaborn        | 0.12+   |
| tqdm           | 4.65+   |

### Installation rapide

```bash
pip install torch torchvision scikit-learn xgboost lightgbm pandas numpy matplotlib seaborn tqdm


## Exécution du notebook

1. Placer le notebook (`Pipeline Complet TP 2.ipynb`) dans le même répertoire que le dossier `data/`.
2. Ouvrir le notebook avec VSCode.
3. Exécuter les cellules dans l’ordre (de haut en bas).
4. Si vous disposez d’un GPU compatible CUDA, PyTorch l’utilisera automatiquement. Sinon, l’exécution se fera sur CPU (plus lent).

## Temps d’exécution approximatif

Les temps ci-dessous ont été mesurés sur une machine équipée d’un CPU Intel Core i7 et de 16 Go de RAM. L’utilisation d’un GPU réduit ces temps.

- Entraînement du CNN personnalisé (4 datasets × 20 époques) : 1h
- Extraction de caractéristiques (6 architectures) + classification (5 classifieurs) sur 4 datasets : 50 min
- Exécution complète du notebook** : 2h

**Remarque** : La première exécution peut être plus lente car les modèles pré-entraînés (AlexNet, ResNet, etc.) sont téléchargés automatiquement par torchvision.

## Résultats attendus

À la fin de la partie 1, le notebook génère automatiquement :
- Les courbes d’apprentissage (perte et accuracy) pour le CNN personnalisé.
- Les matrices de confusion pour chaque dataset.

À la fin de la partie 2, le notebook génère automatiquement :
- L'affichage détaillé pour chaque combinaison (architecture + classifieur) : accuracy et F1-score.
- Tableau comparatif complet de toutes les combinaisons testées.
- Indication de la meilleure combinaison (architecture et classifieur) avec son F1-score.

À la fin des deux parties, le notebook génère automatiquement :
- Un graphique à barres comparant les meilleurs F1-scores obtenus en Partie 1 (CNN personnalisé) et en Partie 2 (extraction + ML).
- Ce graphique permet de visualiser rapidement les gains apportés par l’approche par transfert d’apprentissage sur chaque dataset.