# Prédiction de voyelles

> TP 2 SAN - 3ETI - CPE Lyon

**__Auteur:__** **Tristan GUIEU, Nino BELAOUD**

**__Environement Python:__** **MSI CPE**

## Principe de fonctionnement

### Entrainement

Le but de l'entrainement est de calculer le vecteur caractérisant chaque lettre.

Chaque lettre possède un vecteur est de dimension H+1 composée des amplitudes moyennes du fondamentales et des H premier harmonique des signaux de la lettre contenu dans `data/apprentissage`.

-- Mettre formules latex --

Nous obtenons les vecteurs suivants:

-- Resultat latex --

-- Flowchart --

## Prediction

Le but de la prédiction est de trouver à quel lettre un signal inconnu correspond.

Pour cela il faut trouver calculer le score de similitude entre le **vecteur du signal inconnu** et chaque **vecteur des lettres connues**.

La lettre détecté étant celle avec le score de similitude le plus élevé.

Nous avons choisit de définir le score de similitude par le **produit scalaire** entre le **vecteur inconnu normalisé** et le **vecteur connu normalisé**

-- Mettre formule latex --

On choisit ensuite celui avec le plus grand score.

-- Flowchart --