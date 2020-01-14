import numpy as np
import json


with open("models/20200113-100601/002240_final.json", "r") as fp:
    data = json.load(fp)

predictions = np.array(data["predictions"])
labels = np.array(data["labels"])

predicted_two_indices = (predictions == 1).nonzero()[0]
real_two_indices = (labels == 1).nonzero()[0]

print(predicted_two_indices)
print(real_two_indices)
print(len(predicted_two_indices))
print(len(real_two_indices))

tp, fp, tn, fn = 0, 0, 0, 0
for i in range(len(predictions)):
    if i in predicted_two_indices and i in real_two_indices:
        tp += 1
    elif i in predicted_two_indices and i not in real_two_indices:
        fp += 1
    elif i not in predicted_two_indices and i in real_two_indices:
        fn += 1
    else:
        tn += 1

print("TP: ", tp)
print("FP: ", fp)
print("FN: ", fn)
print("TN: ", tn)

precision = tp/(tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print("F1 Score: ", f1)
