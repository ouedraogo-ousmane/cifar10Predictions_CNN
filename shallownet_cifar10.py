# Importation des librairies
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from nn.conv.shallownet import ShallowNet
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from keras.optimizers import SGD
from imutils import paths
import argparse
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10

# Chargement des données CIFAR10
print("[INFO] chargement de CIFAR-10 .....")
((trainX,trainY),(testX,testY)) = cifar10.load_data()

# Transformation des données d'inputs
trainX = trainX.astype("float") / 255
testX = testX.astype("float") / 255

# Encodage des données
encoder = LabelBinarizer()
trainY = encoder.fit_transform(trainY)
testY = encoder.fit_transform(testY)
labelNames = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]

# Initialisation de optimiseur et creation du model
sgd = SGD(0.01)
model = ShallowNet.build(32, 32, 3, classes=10)
model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=["accuracy"])

# Entrainement du modèle
print("[INFO] entrainement du modèle .....")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

# Evaluation du modèle
print("[INFO] evaluation du modèle .....")
predictions = model.predict(testX)
print(classification_report(predictions.argmax(axis=1), testY.argmax(axis=1), target_names=labelNames))


# Visualisation graphique
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"],label="train_loss")
plt.plot(np.arange(0,100), H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,100), H.history["accuracy"], label="accuracy")
plt.plot(np.arange(0,100), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("# Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.show()