#  Inspection Routière par IA : Détection Automatisée de Fissures

Ce projet propose une solution de vision par ordinateur basée sur l'architecture **Attention U-Net** pour identifier et évaluer la sévérité des dégradations routières (fissures, nids-de-poule) en contexte tropical (Cameroun).
## Dataset (Disponible sur Kaggle)
Pour des raisons de taille, le jeu de données complet (images locales annotées + CrackForest Dataset) n'est pas hébergé sur GitHub. 
👉 **[Télécharger le Dataset Complet sur Kaggle](https://www.kaggle.com/datasets/anitadongmo/african-roads-crack-detection-dataset)**

##  Fonctionnalités
* **Extraction de contexte :** Entraînement initial sur le CrackForest Dataset (CFD).
* **Adaptation au domaine local :** Fine-tuning sur un jeu de données camerounais (latérite, eau stagnante).
* **Interface Décisionnelle :** Déploiement via Gradio générant des rapports de sévérité.

##  Structure du Projet
* `/notebooks/` : Contient le code d'entraînement complet.
* `/data/` : Échantillons d'images pour tester le modèle.
* `/models/` : Poids du modèle entraîné (.pth).
* `app.py` : Script de lancement de l'interface utilisateur (Gradio).


##  Comment tester l'application ?

1. Clonez ce dépôt :
   ```bash
   git clone [https://github.com/AnitaBelviane/Inspection-Routiere-IA.git](https://github.com/votre-nom/Inspection-Routiere-Ia.git)
   ```
2. Installez les dépendances :
 ```
pip install -r requirements.txt
 ```
3. Lancez l'interface Gradio :
```
python app.py
```

## Auteurs
* **DONGMO TCHOUMENE Anita B.**
* **BOKOU-BOUNA Ange-Larissa**
* **JIATSA DONHACHI Rommel J.**

*Projet réalisé dans le cadre du Master 1 Data Science - Université de Yaoundé I (2025-2026).*
