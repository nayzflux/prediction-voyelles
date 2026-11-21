import numpy as np
import numpy as np
import msicpe.san as san
from pathlib import Path
from lib import *

dossier = Path("./data/test")

letters = Tri(dossier)

# Parcourir toutes les lettres
for letter in letters:
    print("-------------------------------------")

    # Pour chaque signal
    for fichier in letters[letter]:
        # Charger le signal
        # Extraire frequence fondamentale et listes des amplitudes des harmoniques
        freq_fondamentales, amplitudes = Signal(fichier, [], [], None, True)

        freq_fondamentale = freq_fondamentales[0]
        amplitudes = amplitudes[0]

        scores = Predict(amplitudes)
        poids = PoidsSoftmax(scores)

        print("========================")

        prediction = Classify(poids)

        confiance = format(poids[prediction] * 100, ".2f")

        if letter == prediction:
            print("✅")
        else:
            print("❌")

        print(f"Réelle : {letter}")
        print(f"Prédiction : {prediction}")
        print(f"Confiance relative (softmax) : {confiance}%")

