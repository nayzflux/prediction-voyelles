import numpy as np
import numpy as np
import msicpe.san as san
from pathlib import Path

# Nombre d'harmonique
H = 14

# Seuil de la detection du fondamentale
seuil = 0.3

# Calcule score
# PSN : Produit Scalaire avec Vecteur Normalisé
# PS : Produit Scalaire
# NE : Norme Euclidienne
# MSE : Erreur Quadratique Moyenne

score_type = "NE"

vecs = {'a': [np.float32(0.99267256), np.float32(0.95044786), np.float32(0.7305962), np.float32(0.80916995), np.float32(0.7818513), np.float32(0.87888974), np.float32(0.8140052), np.float32(0.7859487), np.float32(0.72547275), np.float32(0.66337895), np.float32(0.7659803), np.float32(0.5816474), np.float32(0.42554927), np.float32(0.31924766), np.float32(0.23280911)], 'e': [np.float32(0.9708886), np.float32(0.9713222), np.float32(0.8103081), np.float32(0.9111147), np.float32(0.6494632), np.float32(0.5345234), np.float32(0.50967103), np.float32(0.49420872), np.float32(0.40528193), np.float32(0.47410646), np.float32(0.47640872), np.float32(0.36014467), np.float32(0.26742417), np.float32(0.29906645), np.float32(0.22626229)], 'i': [np.float32(0.98779315), np.float32(0.95045453), np.float32(0.6359798), np.float32(0.39834732), np.float32(0.2264533), np.float32(0.21200545), np.float32(0.1358447), np.float32(0.04303953), np.float32(0.01995408), np.float32(-0.034380574), np.float32(-0.010712976), np.float32(-0.009577533), np.float32(0.004550211), np.float32(0.059550177), np.float32(0.15248403)], 'o': [np.float32(0.9231264), np.float32(0.9827339), np.float32(0.97002494), np.float32(0.79063225), np.float32(0.57761306), np.float32(0.59391373), np.float32(0.3628173), np.float32(0.22879446), np.float32(0.14405213), np.float32(0.10148162), np.float32(0.077704385), np.float32(0.04261342), np.float32(0.034752853), np.float32(0.018442163), np.float32(0.016641542)], 'u': [np.float32(0.9989991), np.float32(0.9375993), np.float32(0.62855315), np.float32(0.41422352), np.float32(0.32762104), np.float32(0.23916325), np.float32(0.12206634), np.float32(0.09481355), np.float32(0.10798422), np.float32(0.10435078), np.float32(0.20470314), np.float32(0.32042783), np.float32(0.28430885), np.float32(0.2706696), np.float32(0.21279886)]}

def SetVecs(new_vecs: dict[str, list[np.float32]]):
    global vecs
    vecs = new_vecs

def SetScoreType(type: str):
    global score_type
    score_type = type

def SetH(h: int):
    global H
    H = h

def SetSeuil(s: float):
    global seuil
    seuil = s

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
    f1 = san.detect_fondamentale(mod/np.max(mod), nu, seuil)
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

def NormeEuclidienne(vec1, vec2):
    return float(np.linalg.norm(
        np.asarray(vec1) - np.asarray(vec2)
    ))

def ErreurQuadratiqueMoyenne(vec1, vec2):
    return np.mean((vec1 - vec2)**2)


def Normalize(vec):
    norme = np.sqrt(ProduitScalaire(vec, vec))
    return vec / norme

def Predict(amplitudes):
    scores = {}
    for letter, vec_connu in vecs.items():
        # Calculer le score de similitude
        if score_type == "PSN":
            scores[letter] = ProduitScalaire(Normalize(vec_connu), Normalize(amplitudes))
        elif score_type == "PS":
            scores[letter] = ProduitScalaire(vec_connu, amplitudes)
        elif score_type == "NE":
            scores[letter] = -NormeEuclidienne(amplitudes, vec_connu)
        elif score_type == "MSE":
            scores[letter] = -ErreurQuadratiqueMoyenne(amplitudes, vec_connu)

    return scores

def PoidsSoftmax(scores):
    valeurs = np.asarray(list(scores.values()), dtype=float)
    if valeurs.size == 0 or not np.isfinite(valeurs).all():
        raise ValueError("La softmax exige au moins un score et des scores tous finis.")

    # Soustraire le maximum stabilise le calcul des exponentielles.
    poids = np.exp(valeurs - valeurs.max())
    poids /= poids.sum()
    return dict(zip(scores, poids))

def Classify(scores):
    score_max = 0
    guess = ""
    for letter in scores.keys():
        score = scores[letter]
        if score >= score_max:
            score_max = score
            guess = letter
    return guess