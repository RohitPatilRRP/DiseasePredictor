import csv
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

disease = []
symptom = []


with open('Dataset.csv', 'r') as f:
     reader = csv.reader(f, delimiter=',')
     for row in reader:
        disease.append(row[0])
        symptom.append(row[1])

clf = tree.DecisionTreeClassifier()

try:
	clf= clf.fit(symptom, disease)
except ValueError:
    print("error on line")
	
print(clf.predict(['feeling hopeless']))
# model  = GaussianNB()
# model.fit(symptom,disease)
# print(model.predict(["agitation"],["exhaustion"]))

# print(symptom)