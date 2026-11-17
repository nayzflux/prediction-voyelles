import numpy as np
import numpy as np
import msicpe.san as san
from pathlib import Path

"""
Constantes
"""
seuil = 0.9
H = 20
dossier = Path("./data/test")
# Dictionnaire des vecteurs trouvés pendant l'entrainement
# vecs = {'a': [np.float32(0.99267256), np.float32(0.95044786), np.float32(0.7305962), np.float32(0.80916995), np.float32(0.7818513), np.float32(0.87888974), np.float32(0.8140052), np.float32(0.7859487), np.float32(0.72547275), np.float32(0.66337895), np.float32(0.7659803), np.float32(0.5816474), np.float32(0.42554927), np.float32(0.31924766), np.float32(0.23280911), np.float32(0.18744649), np.float32(0.15079428), np.float32(0.19357136), np.float32(0.13383953), np.float32(0.17017466), np.float32(0.20891285)], 'e': [np.float32(0.9708886), np.float32(0.9713222), np.float32(0.8103081), np.float32(0.9111147), np.float32(0.6494632), np.float32(0.5345234), np.float32(0.50967103), np.float32(0.49420872), np.float32(0.40528193), np.float32(0.47410646), np.float32(0.47640872), np.float32(0.36014467), np.float32(0.26742417), np.float32(0.29906645), np.float32(0.22626229), np.float32(0.24842057), np.float32(0.22837368), np.float32(0.21968004), np.float32(0.21687213), np.float32(0.2083003), np.float32(0.23020121)], 'i': [np.float32(0.98779315), np.float32(0.95045453), np.float32(0.6359798), np.float32(0.39834732), np.float32(0.2264533), np.float32(0.21200545), np.float32(0.1358447), np.float32(0.04303953), np.float32(0.01995408), np.float32(-0.034380574), np.float32(-0.010712976), np.float32(-0.009577533), np.float32(0.004550211), np.float32(0.059550177), np.float32(0.15248403), np.float32(0.122236036), np.float32(0.018672634), np.float32(-0.0060028927), np.float32(-0.017602008), np.float32(-0.03539036), np.float32(0.04755984)], 'o': [np.float32(0.9433061), np.float32(0.94629705), np.float32(0.8812683), np.float32(0.67352635), np.float32(0.47385675), np.float32(0.49871746), np.float32(0.31884784), np.float32(0.18432224), np.float32(0.120058835), np.float32(0.07766334), np.float32(0.097653806), np.float32(0.038675524), np.float32(0.06606811), np.float32(0.012193746), np.float32(0.014555059), np.float32(0.010326604), np.float32(-0.0182327), np.float32(-0.026162941), np.float32(0.052595653), np.float32(0.101860024), np.float32(0.06661477)], 'u': [np.float32(0.9989991), np.float32(0.9375993), np.float32(0.62855315), np.float32(0.41422352), np.float32(0.32762104), np.float32(0.23916325), np.float32(0.12206634), np.float32(0.09481355), np.float32(0.10798422), np.float32(0.10435078), np.float32(0.20470314), np.float32(0.32042783), np.float32(0.28430885), np.float32(0.2706696), np.float32(0.21279886), np.float32(0.15733898), np.float32(0.045500897), np.float32(-0.037463754), np.float32(0.018077735), np.float32(-0.0033590929), np.float32(0.075389534)]}
vecs = {'a': [np.float32(0.99267256), np.float32(0.95044786), np.float32(0.7305962), np.float32(0.80916995), np.float32(0.7818513), np.float32(0.87888974), np.float32(0.8140052), np.float32(0.7859487), np.float32(0.72547275), np.float32(0.66337895), np.float32(0.7659803), np.float32(0.5816474), np.float32(0.42554927), np.float32(0.31924766), np.float32(0.23280911), np.float32(0.18744649), np.float32(0.15079428), np.float32(0.19357136), np.float32(0.13383953), np.float32(0.17017466), np.float32(0.20891285)], 'e': [np.float32(0.9708886), np.float32(0.9713222), np.float32(0.8103081), np.float32(0.9111147), np.float32(0.6494632), np.float32(0.5345234), np.float32(0.50967103), np.float32(0.49420872), np.float32(0.40528193), np.float32(0.47410646), np.float32(0.47640872), np.float32(0.36014467), np.float32(0.26742417), np.float32(0.29906645), np.float32(0.22626229), np.float32(0.24842057), np.float32(0.22837368), np.float32(0.21968004), np.float32(0.21687213), np.float32(0.2083003), np.float32(0.23020121)], 'i': [np.float32(0.98779315), np.float32(0.95045453), np.float32(0.6359798), np.float32(0.39834732), np.float32(0.2264533), np.float32(0.21200545), np.float32(0.1358447), np.float32(0.04303953), np.float32(0.01995408), np.float32(-0.034380574), np.float32(-0.010712976), np.float32(-0.009577533), np.float32(0.004550211), np.float32(0.059550177), np.float32(0.15248403), np.float32(0.122236036), np.float32(0.018672634), np.float32(-0.0060028927), np.float32(-0.017602008), np.float32(-0.03539036), np.float32(0.04755984)], 'o': [np.float32(0.9433061), np.float32(0.94629705), np.float32(0.8812683), np.float32(0.67352635), np.float32(0.47385675), np.float32(0.49871746), np.float32(0.31884784), np.float32(0.18432224), np.float32(0.120058835), np.float32(0.07766334), np.float32(0.097653806), np.float32(0.038675524), np.float32(0.06606811), np.float32(0.012193746), np.float32(0.014555059), np.float32(0.010326604), np.float32(-0.0182327), np.float32(-0.026162941), np.float32(0.052595653), np.float32(0.101860024), np.float32(0.06661477)], 'u': [np.float32(0.9989991), np.float32(0.9375993), np.float32(0.62855315), np.float32(0.41422352), np.float32(0.32762104), np.float32(0.23916325), np.float32(0.12206634), np.float32(0.09481355), np.float32(0.10798422), np.float32(0.10435078), np.float32(0.20470314), np.float32(0.32042783), np.float32(0.28430885), np.float32(0.2706696), np.float32(0.21279886), np.float32(0.15733898), np.float32(0.045500897), np.float32(-0.037463754), np.float32(0.018077735), np.float32(-0.0033590929), np.float32(0.075389534)]}
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

def ProduitScalaire(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError(f"vec1 ({len(vec1)}) et vec2 ({len(vec2)}) doivent être de la meme dimension")
    
    sum = 0

    for i in range(len(vec1)):
        sum += vec1[i] * vec2[i]

    return sum

def Normalize(vec):
    norme = np.sqrt(ProduitScalaire(vec, vec))
    return vec / norme

def Predict(amplitudes):
    scores = {}
    for letter in letters:
        vec_connu = vecs[letter]
        # Le score de similitude entre le vecteur connu de la lettre et le vecteur inconnu est le produit scalaire
        scores[letter] = ProduitScalaire(Normalize(vec_connu), Normalize(amplitudes))
    return scores

def Classify(scores):
    score_max = 0
    guess = ""
    for letter in scores.keys():
        score = scores[letter]
        if score >= score_max:
            score_max = score
            guess = letter
    return guess


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

        print("========================")
        # print(scores)

        prediction = Classify(scores)

        # Meilleur score
        confiance = format(scores[prediction] * 100, ".2f")

        if letter == prediction:
            print("✅")
        else:
            print("❌")

        print(f"Réelle : {letter}")
        print(f"Prédiction : {prediction}")
        print(f"Confiance : {confiance}%")

