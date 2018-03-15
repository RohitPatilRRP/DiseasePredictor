import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

#with open("Dataset.csv", "r") as file: 
 #   for i in file: #FOR EACH LINE
  #     lines = file._next_().split(",") 
   #    model.fit(lines[1],lines[0])
	   
	   
data = np.genfromtxt("Dataset.csv", dtype=None, delimiter=',', names=True)
model.fit(data[1],data[0])
predicted = model.predict([["vomit"]])
print(predicted)