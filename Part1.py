import numpy as np

f = open("OnlineNewsPopularity.csv", "r")
data = []
totaldata = len(data)
for line in f:
	singleline = line.split(",")
	data.append(singleline)

X = data[(totaldata/5:totaldata]
XT = np.transpose(X)
Y = data[X, 60]
first = np.dot(XT, X)
second = np.dot(XT, Y)
inverse = np.linalg.inverse(first)
weights = np.dot(inverse, second)