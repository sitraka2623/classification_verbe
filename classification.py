import numpy as np
import tensorflow as tf

# Données d'entraînement : verbes et leurs groupes
verbes = ["aimer", "finir", "prendre", "parler", "choisir", "manger", "courir", "vendre", "ouvrir"]
groupes = [1, 2, 3, 1, 2, 1, 2, 3, 2]  # 1er, 2e ou 3e groupe

# Extraction des caractéristiques des verbes
def extraire_caracteristiques(verbes):
    caracteristiques = []
    for verbe in verbes:
        longueur = len(verbe)
        terminaison_er = 1 if verbe.endswith("er") else 0
        terminaison_ir = 1 if verbe.endswith("ir") else 0
        terminaison_re = 1 if verbe.endswith("re") else 0
        presence_voyelle = sum(1 for c in verbe if c in "aeiou")
        caracteristiques.append([longueur, terminaison_er, terminaison_ir, terminaison_re, presence_voyelle])
    return np.array(caracteristiques, dtype=np.float32)


X = extraire_caracteristiques(verbes)
Y = tf.keras.utils.to_categorical(groupes - 1, num_classes=3)  


model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu", input_shape=(5,)),  
    tf.keras.layers.Dense(4, activation="relu"), 
    tf.keras.layers.Dense(3, activation="softmax") 
])

# Compilation et entraînement
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X, Y, epochs=500, verbose=0)

# Fonction de classification
def classifier_verbe(verbe):
    X_test = extraire_caracteristiques([verbe])
    prediction = model.predict(X_test)
    groupe = np.argmax(prediction) + 1
    return f"{verbe} appartient au {groupe}ᵉ groupe"

# Test du modèle
print(classifier_verbe("marcher"))   # 1er groupe
print(classifier_verbe("grandir"))   # 2e groupe
print(classifier_verbe("répondre"))  # 3e groupe
