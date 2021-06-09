import os

with open("map.txt","r") as f:
    lines = f.readlines()


w = []

for line in lines:
    w.append(line.split()[-2])

w = [float(i) for i in w]
print(w)
