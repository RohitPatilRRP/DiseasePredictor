import json,os,sys,re
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier


diseaseclassifier = Trainer(tokenizer) 
with open("Dataset.csv", "r") as file:
    for i in file: #FOR EACH LINE
       lines = file.readline().split(",") 
       diseaseclassifier.train(lines[1],  lines[0]) 
diseaseclassifier = Classifier(diseaseclassifier.data, tokenizer)
txt = input("enter symptomA symptomB symptomC")
classification = diseaseclassifier.classify(txt) 
print(classification[0]) 
