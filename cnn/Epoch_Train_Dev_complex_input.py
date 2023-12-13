import sys
import re
import matplotlib.pyplot as plt

epoch = []
train = []
dev = []

with open("epoch_data.txt", 'r') as f:
    for line in f.readlines():
        if "Epoch " in line:
            epoch.append(float(re.search(r"Epoch \d+", line).group().split("Epoch ")[1]))
            train.append(float(re.search(r"Train F1: \d.\d+", line).group().split("Train F1: ")[1]))
            dev.append(float(re.search(r"Dev F1: \d.\d+", line).group().split("Dev F1: ")[1]))
        print(line, end='')

fig, ax = plt.subplots()
ax.scatter(epoch, train, c="red", label="train f1")
ax.scatter(epoch, dev, c="blue", label="dev f1")

ax.legend()
ax.grid(True)
ax.set_xlabel("Epoch")
ax.set_ylabel("f1")
plt.show()