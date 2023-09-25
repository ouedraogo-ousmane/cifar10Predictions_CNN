# Projet de Reconnaissance d'Images avec 10 Classes (CIFAR-10)

Ce projet utilise un réseau de neurones convolutionnels (CNN) pour la reconnaissance d'images appartenant à 10 catégories différentes, tirées de la base de données CIFAR-10.

## Introduction

La base de données CIFAR-10 est un ensemble de données bien connu en vision par ordinateur, composé de 60 000 images réparties en 10 classes équilibrées.

## Dataset

Nous avons utilisé l'ensemble de données CIFAR-10 de Keras, qui comprend des images de 10 classes différentes : avions, automobiles, oiseaux, chats, cerfs, chiens, grenouilles, chevaux, navires et camions.

## Architecture du Modèle

Le modèle CNN comprend des couches de convolution, de pooling, des couches denses (fully connected) et une couche de sortie. Cette architecture est spécialement conçue pour extraire des caractéristiques discriminantes des images.

## Entraînement et Évaluation

Le modèle a été entraîné sur l'ensemble d'entraînement CIFAR-10 avec une validation croisée pour éviter le surajustement. Nous avons utilisé l'optimiseur Adam et la fonction de perte de l'entropie croisée catégorielle.

Le modèle a été évalué sur l'ensemble de test pour mesurer ses performances.

## Utilisation du Code

1. Assurez-vous d'installer les dépendances nécessaires en exécutant `pip install -r requirements.txt`.
2. Entraînez le modèle en exécutant `python shallownet_cifar10.py`. Le modèle entraîné sera enregistré sous forme de fichier.


## Résultats

Le modèle a atteint une précision de X% sur l'ensemble de test, démontrant son efficacité dans la classification des images de 10 catégories différentes.

## Remarques

N'hésitez pas à explorer et à adapter ce projet pour vos propres expérimentations en reconnaissance d'images.

## Auteur

OUEDRAOGO Ousmane

oueo5587@gmail.com


