🔠 Classification des verbes français en groupes

Ce projet utilise 
Python 3.13.4, 
NumPy, 
TensorFlow
 pour entraîner un modèle de réseau de neurones afin de classifier les verbes français en premier, deuxième ou troisième groupe.

 📌 Description
L'objectif est d'entraîner un modèle Perceptron multicouche qui, à partir des caractéristiques d'un verbe (longueur, terminaison, présence de voyelles...), détermine son groupe grammatical.

 🛠 Technologies utilisées
- Python 3.13.4
- TensorFlow
- NumPy

⚙️ Fonctionnement
1. Extraction des caractéristiques du verbe :
   - Longueur du verbe
   - Terminaison (`-er`, `-ir`, `-re`)
   - Présence de voyelles
2. Entraînement d’un modèle avec :
   - 5 perceptrons dans la première couche
   - 4 perceptrons intermédiaires
   - 3 perceptrons pour la classification finale
3. Prédiction du groupe d’un verbe donné.

 🚀 Installation et exécution
 1️⃣ Cloner le projet
sh
git clone https://github.com/sitraka2623/classification_verbe.git
cd classification_verbe
