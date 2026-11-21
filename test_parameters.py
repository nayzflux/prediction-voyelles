from pathlib import Path

import numpy as np
from plotly.subplots import make_subplots

import lib

dossier_apprentissage = Path("./data/apprentissage")
apprentissage = lib.Tri(dossier_apprentissage)
dossier_test = Path("./data/apprentissage")
test = lib.Tri(dossier_test)
methodes = ("PSN", "PS", "NE", "MSE")

def amplitudes(fichier):
    _, vecteurs = lib.Signal(fichier, [], [])
    return np.asarray(vecteurs[0])


def mesurer(h, s):
    lib.SetH(h)
    lib.SetSeuil(s)
    lib.SetVecs({
        voyelle: np.mean([amplitudes(f) for f in fichiers], axis=0)
        for voyelle, fichiers in apprentissage.items()
    })
    signaux = [(voyelle, amplitudes(f)) for voyelle, fichiers in test.items() for f in fichiers]

    resultats = {}
    for methode in methodes:
        lib.SetScoreType(methode)
        correctes = 0
        confiances = []
        for voyelle, signal in signaux:
            poids = lib.PoidsSoftmax(lib.Predict(signal))
            prediction = lib.Classify(poids)
            correctes += prediction == voyelle
            confiances.append(poids[prediction])
        precision = 100 * correctes / len(signaux)
        confiance = 100 * np.mean(confiances)
        resultats[methode] = (precision, confiance)
    return resultats


valeurs_h = list(range(1, 21))
valeurs_s = [i / 10 for i in range(1, 10)]
resultats_h = [mesurer(h, 0.3) for h in valeurs_h]
resultats_s = [mesurer(13, s) for s in valeurs_s]

fig = make_subplots(
    rows=2,
    cols=2,
    shared_xaxes=True,
    subplot_titles=(
        "Précision selon H (s = 0,3)",
        "Précision selon s (H = 13)",
        "Confiance moyenne selon H (s = 0,3)",
        "Confiance moyenne selon s (H = 13)",
    ),
)

balayages = [(valeurs_h, resultats_h), (valeurs_s, resultats_s)]
styles = [("#0072B2", "solid"), ("#D55E00", "dot"), ("#009E73", "solid"), ("#A54C87", "dash")]
for colonne, (valeurs, resultats) in enumerate(balayages, start=1):
    for methode, (couleur, tiret) in zip(methodes, styles):
        for ligne in (1, 2):
            fig.add_scatter(
                x=valeurs,
                y=[resultat[methode][ligne - 1] for resultat in resultats],
                name=methode,
                legendgroup=methode,
                showlegend=ligne == 1 and colonne == 1,
                mode="lines+markers",
                line=dict(color=couleur, dash=tiret),
                row=ligne,
                col=colonne,
            )

fig.update_xaxes(title_text="H", row=2, col=1, dtick=1)
fig.update_xaxes(title_text="Seuil s", row=2, col=2, dtick=0.1)
fig.update_yaxes(title_text="Précision (%)", row=1, col=1)
fig.update_yaxes(title_text="Confiance moyenne (%)", row=2, col=1)
fig.show()
