from lib import Classify, PoidsSoftmax, Predict, Signal
from plotly.subplots import make_subplots

import numpy as np
import msicpe.san as san
import plotly.graph_objects as go

file_name = input("Entrer le chemin du fichier à analyser : \n")

# Créer deux graphiques
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=("Signal temporel", "Spectre de Fourier")
)

freq_fondamentales, amplitudes = Signal(
    file_name, [], [], pI=1, pFig=fig, pAffichage=True
)

freq_fondamentale = freq_fondamentales[0]
amplitudes = amplitudes[0]

scores = Predict(amplitudes)
poids = PoidsSoftmax(scores)

print("========================")

prediction = Classify(poids)

confiance = format(poids[prediction] * 100, ".2f")

print(f"Frequence fondamentale : {freq_fondamentale}")
print(f"Fichier : {file_name}")
print(f"Prédiction : {prediction}")
print(f"Confiance relative (softmax) : {confiance}%")

print("-------")
print("Scores")

for v in scores.keys():
    print(f"{v} : {scores[v]}")

print("-------")
print("Vecteur :")

for a in amplitudes:
    print(a)

print("-------")

print("========================")

# Titres et axes
fig.update_layout(
    title=f"Analyse du signal : {file_name}",
    height=700
)

fig.update_xaxes(title_text="Temps (s)", row=1, col=1)
fig.update_yaxes(title_text="Amplitude", row=1, col=1)

fig.update_xaxes(title_text="Fréquence (Hz)", row=2, col=1)
fig.update_yaxes(title_text="Amplitude du spectre", row=2, col=1)

# Afficher les graphiques
fig.show()