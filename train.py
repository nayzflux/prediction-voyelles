"""
Importation des bibliothèques
"""

import numpy as np
import msicpe.san as san
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from lib import *

dossier = Path("./data/apprentissage")

letters = Tri(dossier)

# Parcourir toutes les lettres
for letter in letters:
    # On intialise un vecteur vide pour chaque lettre
    vecs[letter] = []

    # Créer deux graphiques
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Signal temporel", "Spectre de Fourier")
    )

    i = 1

    # Liste contient les frequences fondamentales de tous les signaux
    freq_fondamentales = []

    # Liste qui contient la listes des amplitudes des harmoniques de tous les signaux
    amplitudes = []

    # Pour chaque signal
    for fichier in letters[letter]:
        # Charger le signal
        # Extraire frequence fondamentale et listes des amplitudes des harmoniques
        freq_fondamentales, amplitudes = Signal(fichier, freq_fondamentales, amplitudes, fig, True)

        i+=1

    # Trouver la frequence fondamentale moyenne
    # Trouver les amplitudes moyennes des harmoniques
    frequence_fondamentale_moyenne, amplitudes_moyenne = Moyennes(freq_fondamentales, 
                                                                  amplitudes, 
                                                                  len(letters[letter]))

    # On definit le vecteur de la lettre comme la listes des amplitudes moyennes
    vecs[letter] = amplitudes_moyenne

    # Titres et axes
    fig.update_layout(
        title=f"Analyse du signal : {letter}",
        height=700
    )

    fig.update_xaxes(title_text="Temps (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)

    fig.update_xaxes(title_text="Fréquence (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude du spectre", row=2, col=1)

    # Afficher les graphiques
    fig.show()

# Afficher tous les vecteurs
for letter in letters:
    print("=======================")
    print(f"Lettre : {letter}")
    for amplitude in vecs[letter]:
        print(amplitude)
    print(vecs)
    print("=======================")

