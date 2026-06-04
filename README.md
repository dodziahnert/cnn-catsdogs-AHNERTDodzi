# cnn-catsdogs-AHNERTDodzi

## Objectifs du projet

Ce projet implémente et compare deux approches de Deep Learning pour la classification binaire d'images (chats contre chiens) en utilisant PyTorch. L'objectif principal est de mettre en évidence les différences de performance et d'apprentissage entre :

* Un réseau de neurones convolutif (CNN) créé et entraîné de zéro (From Scratch).
* Un modèle pré-entraîné utilisant l'apprentissage par transfert (Transfer Learning avec MobileNet_V3_Large).

## Fonctionnalités Principales

* Prétraitement et Augmentation des Données : Redimensionnement (224x224), rotations aléatoires, retournements horizontaux et normalisation via torchvision.transforms.v2.
* Répartition Automatique : Séparation des données en ensembles d'entraînement (80%), de validation (10%) et de test (10%).
* Optimisation des Hyperparamètres : Utilisation d'Optuna pour trouver automatiquement le meilleur taux d'apprentissage (learning rate), le meilleur optimiseur (Adam vs SGD) et le nombre optimal de couches à geler pour le Transfer Learning.
* Suivi des Expérimentations : Intégration de MLflow en local pour journaliser les paramètres, les métriques (Loss, Accuracy, Precision, Recall) et les artefacts (matrices de confusion, visualisations des prédictions, modèles sauvegardés).

Évaluation et Visualisation : Génération de matrices de confusion binaire, calcul de la précision par classe, et création de graphiques comparatifs (Loss et Accuracy) entre les deux modèles.

## Technologies Utilisées

* Langage : **Python (3.13)**
* Framework Deep Learning : **PyTorch & Torchvision**
* Métriques : **TorchMetrics, Scikit-learn**
* Optimisation : **Optuna**
* MLOps / Tracking : **MLflow**
* Manipulation de Données & Visualisation : **Pandas, NumPy, Matplotlib, Seaborn, tqdm.**

## Prérequis

Assurez-vous d'avoir Python installé ainsi que les bibliothèques nécessaires. Vous pouvez générer un environnement virtuel et installer les dépendances listées dans un requirements.txt:

```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

*Note : L'utilisation d'un GPU (CUDA) est fortement recommandée pour accélérer l'entraînement. le notebook effectue cette vérification et l'utilise si disponible.*

## Structure des Données

Les données doivent être placées dans le répertoire racine du projet. Une cellule du notebook se charge de réorganiser automatiquement l'arborescence brute en un format lisible par torchvision.datasets.ImageFolder.

Format d'origine attendu :
Placez vos dossiers téléchargés sous le chemin suivant : data/Cat_Dog_data/

Format généré par le script :
Le code va extraire et déplacer les images dans un nouveau dossier structuré (data/Cat_Dog_ordered/) :

```plaintext
data/
└── Cat_Dog_ordered/
    ├── cat/
    │   ├── image_01.jpg
    │   ├── image_02.jpg
    │   └── ...
    └── dog/
        ├── image_01.jpg
        ├── image_02.jpg
        └── ...
```

Les données sont ensuite réparties dynamiquement lors de l'exécution :

* **80% Entraînement** pour permettre l'entrainement du modèle
* **10% Validation** pour la validation
* **10% Test** pour les tests des modèles finaux et la comparaison entre les deux modèles.

## Paramètres d'entraînement et Exécution

L'entraînement est piloté par l'outil d'optimisation Optuna suivi d'un entraînement final journalisé avec MLflow. Les paramètres globaux sont : `Batch size = 32`, `Epochs = 20`.

1. Entraînement From Scratch

    Exécution via la fonction fromscratch() dans le notebook.

    * Architecture : CNN de 3 blocs (Conv2d, BatchNorm2d, ReLU, MaxPool2d) suivi d'un classifieur linéaire.
    * Batch Normalization (BN) : Appliquée après chaque couche convolutive (32, 64, puis 128 filtres).
    * Dropout : 0.5 dans le classifieur linéaire pour limiter le surapprentissage.
    * Optimiseur & Learning Rate (LR) : Déterminés par Optuna (Choix entre Adam et SGD ; LR testés : 1e-4, 1e-3, 1e-2).
    * Scheduler : StepLR (step_size=1, gamma=0.7) pour réduire le LR au fil des époques.

2. Entraînement par Transfer Learning

    Exécution via la fonction transfert() dans le notebook.

    * Modèle de base : MobileNet_V3_Large pré-entraîné sur ImageNet.
    * Gel des couches (Freezing) : Le nombre de couches gelées (1 à 8) dans la partie features est déterminé par Optuna. Le reste est "fine-tuné" (entraîné).
    * Classifieur : La dernière couche linéaire du modèle est remplacée pour s'adapter à 2 classes.
    * Optimiseur & Learning Rate : Optuna sélectionne le meilleur combo (Adam/SGD, LR : 1e-4 à 1e-2).
    * Scheduler : StepLR (step_size=1, gamma=0.7).

## Utilisation

1. Préparation : Placez vos données brutes dans le dossier data/Cat_Dog_data. Le script se chargera de les réorganiser.
2. Lancement de MLflow : Pour visualiser vos entraînements en temps réel, ouvrez un terminal et lancez l'interface MLflow (assurez-vous que le port 5000 de ovtre machine soit disponible) :

    > mlflow ui

    Rendez-vous ensuite sur [http://localhost:5000](http://localhost:5000) dans votre navigateur.

3. Exécution : Lancez les cellules du notebook séquentiellement pour :

    * Préparer les DataLoaders.
    * Lancer l'étude Optuna pour le modèle "Scratch".
    * Entraîner le modèle "Scratch" final et journaliser les résultats sur MLflow.
    * Lancer l'étude Optuna pour le modèle "Transfer Learning" (MobileNetV3).
    * Entraîner le modèle "Transfer Learning" final.
    * Afficher les graphiques comparatifs finaux.

## Résultats et évaluation

À la fin de l'exécution, le projet génère :

* Des fichiers de modèles sauvegardés (`best_scratch_model.pth`, `best_transfer_model.pth`).
* Des images de matrices de confusion (`confusion_matrix_scratch.png`, `confusion_matrix_transfer.png`).
* Des planches de visualisation des prédictions correctes et incorrectes.
* Un tableau comparatif des métriques (Accuracy, Precision, Recall) résumant la performance des deux approches sur les données de test non vues.

## Commandes pour évaluer / recharger le modèle

Pendant l'entraînement, les meilleurs modèles (basés sur la validation loss) sont sauvegardés localement. Pour recharger un modèle et l'évaluer sans relancer l'entraînement, vous pouvez utiliser les commandes PyTorch suivantes dans une nouvelle cellule :

Recharger le modèle Scratch :

```python
modele_scratch = CNNScratch().to(device)
modele_scratch.load_state_dict(torch.load("best_scratch_model.pth"))
modele_scratch.eval()

# Évaluation
cm, test_acc, precision, recall = test_and_confusion(modele_scratch, data_loaders["test"])
```

Recharger le modèle Transfer Learning :

```python
modele_transfert = models.mobilenet_v3_large(weights=None)
modele_transfert.classifier[3] = nn.Linear(modele_transfert.classifier[3].in_features, 2)
modele_transfert = modele_transfert.to(device)

modele_transfert.load_state_dict(torch.load("best_transfer_model.pth"))
modele_transfert.eval()

# Évaluation
cm, test_acc, precision, recall = test_and_confusion(modele_transfert, data_loaders["test"])
```

Le script génère automatiquement les courbes via matplotlib (qui sont également enregistrées dans MLflow) :

* Loss : Courbes de la perte de validation au fil des 20 époques (Scratch vs Transfert).
* Accuracy : Courbes de la précision de validation au fil des 20 époques (Scratch vs Transfert).
* Matrices de confusion : confusion_matrix_scratch.png et confusion_matrix_transfer.png.

## analyse comparative

Un modèle entraîné depuis zéro apprend vraiment par lui‑même grâce à la Batch Normalization et à l’augmentation de données. Mais son apprentissage est plus lent : la courbe de Loss montre qu’il peut stagner ou légèrement surapprendre après plusieurs époques, car il doit découvrir seul les caractéristiques de base (bords, textures).

À l’inverse, avec le Transfer Learning (MobileNetV3), l’entraînement est plus rapide et stable. Les poids pré‑entraînés sur ImageNet offrent déjà une bonne représentation visuelle. Le fine‑tuning, en gardant certaines couches gelées, permet au modèle de se spécialiser vite sur la tâche chien/chat, avec une meilleure précision et une perte réduite.

En résumé, le Transfer Learning est bien plus efficace : il converge en moins d’époques, atteint de meilleures performances et généralise mieux sur de nouvelles images.

## Limites et piste d'amélioration

* Le CNN créé possède une architecture très basique (3 blocs). L'ajout de blocs résiduels (façon ResNet) pourrait grandement améliorer ses performances.
* L'entraînement est fixé à 20 époques. Mettre en place un Early Stopping permettrait d'arrêter l'entraînement automatiquement dès que la validation loss remonte, pour sauver du temps de calcul et éviter l'overfitting.
* Au lieu de geler un nombre de couches statique défini par Optuna, on pourrait appliquer une méthode de dégagement progressif des couches intermédiaires avec un taux d'apprentissage décroissant au fur et à mesure que l'on s'enfonce dans le réseau.
