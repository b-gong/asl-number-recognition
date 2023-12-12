import sys
import re
import matplotlib.pyplot as plt

epoch = []
train = []
dev = []

for line in sys.stdin:
    if "Epoch " in line:
        epoch.append(float(re.search(r"Epoch \d+", line).group().split("Epoch ")[1]))
        train.append(float(re.search(r"Train Accuracy: \d.\d+", line).group().split("Train Accuracy: ")[1]))
        dev.append(float(re.search(r"Dev Accuracy: \d.\d+", line).group().split("Dev Accuracy: ")[1]))
    print(line, end='')

fig, ax = plt.subplots()
ax.scatter(epoch, train, c="red", label="train acc")
ax.scatter(epoch, dev, c="blue", label="dev acc")

ax.legend()
ax.grid(True)
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
plt.show()
