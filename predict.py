from lib import Classify, PoidsSoftmax, Predict, Signal

file_name = input("Entrer le chemin du fichier à analyser : \n")

freq_fondamentales, amplitudes = Signal(file_name, [], [], None, True)

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