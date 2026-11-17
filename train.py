"""
Importation des bibliothèques
"""

import numpy as np
import msicpe.san as san
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd

"""
Constantes
"""
seuil = 0.9
H = 20
dossier = Path("./data/apprentissage")
vecs = {}

"""
Fonctions
"""

def Importation(pFichier : dict) -> tuple[dict, dict]:
    son = np.load(pFichier) # dat est un dictionnaire contenant
    s = son['s'] # extraction du signal utile
    t = son['t'] # extraction du vecteur temps associé au signal

    return s, t

def Tri(pDossier):
    letters = {}
    fichiers = sorted(pDossier.glob("*.npz"))

    for fichier in fichiers:
        lettre = fichier.name.split("_")[0]

        if lettre in letters:
            letters[lettre].append(fichier)
        else:
            letters[lettre] = [fichier]

    return letters

def Signal(pFichier, pFrequencesFondamentales, pAmplitudes, pFig = None, pAffichage = False):
    s, t = Importation(pFichier)
    frequencesfondamentales = pFrequencesFondamentales
    amplitudes = pAmplitudes

    nu, S = san.trans_fourier(s, t)
    mod = np.abs(S)**2
    L = np.log10(mod)
    U = L / np.max(L)

    f1 = san.detect_fondamentale(mod/np.max(mod), nu, 0.9)
    frequencesfondamentales.append(f1)

    frequences_pics = Frequences(f1)

    amplitudes.append(Amplitude(U, frequences_pics, nu))

    # Afficher signal et TF
    if pAffichage:
        pFig.add_trace(
            go.Scatter(x=t, y=s, mode="lines", name=f"Signal {i}"),
            row=1, col=1
        )

        pFig.add_trace(
            go.Scatter(x=nu, y=U, mode="lines", name=f"Spectre {i}"),
            row=2, col=1
        )

    return frequencesfondamentales, amplitudes


def Frequences(pF):
    frequences_pics = []
    for h in range(1, H+2):
        frequences_pics.append(h*pF)
    return frequences_pics

def Amplitude(pU, frequences_pics, pNu):
    freq, amplitude = san.detect_pics(pU, frequences_pics, pNu)

    return amplitude

def Moyennes(pFrequencesFondamentales, 
             pAmplitudes,
             pJ):
    freqfmoy = np.mean(pFrequencesFondamentales)

    amplmoy = []

    for i in range(0, H+1):
        values = []
        for j in range(pJ):
            values.append(pAmplitudes[j][i])
        moyenne = np.mean(values)
        amplmoy.append(moyenne)

    return freqfmoy, amplmoy

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

