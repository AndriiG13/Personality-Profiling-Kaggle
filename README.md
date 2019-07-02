# Personality-Profiling-Kaggle
This repository contains the R script for the in-class Kaggle competition on classifying individuals' personality 
based on vlog transcripts. This activity was part of the Behavioural Data Science course at the University of Amsterdam.

The goal was to use text analysis on vlog transcripts of youtubers in order to classify them into 5 different personlity types. 
The text was split into words and then various sentiment features were extracted, such as anger, disgust, fear etc. Then a 
model was made for each persnality type, resulting in 5 models in total. 

Due to the restrictions set by the course the choice of possible machined learning methods was confined to simple regression. 
By making a smiple regression model for each personality, a classfication accuracy of 73% was reached when tested on the test
dataset. 
