import json

import matplotlib.pyplot as plt
import numpy as np

from bbdb_dataset import LABEL_ID_TO_STR


with open("models/20200110-054139/004000.json", "r") as fp:
    labels = json.load(fp)["labels"]

labels = np.array(labels)
labels = labels[(labels != 16) & (labels != 24) & (labels != 29)]

# with open("models/20200112-164221/004440_final.json", "r") as fp:
# with open("models/20200113-080640/002240_final.json", "r") as fp:
with open("models/20200113-100601/002240_final.json", "r") as fp:
    data = json.load(fp)
    binary_predictions = data["predictions"]
    binary_labels = data["labels"]



correct = np.zeros(30)
incorrect = np.zeros(30)
for i, label in enumerate(labels):
    if binary_predictions[i] == binary_labels[i]:
        correct[label] += 1
    else:
        incorrect[label] += 1

fig, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

axs[0].bar(np.arange(30), correct)
axs[0].bar(np.arange(30), incorrect, bottom=correct)

correct, incorrect = correct/(correct + incorrect), incorrect/(correct + incorrect)
axs[1].bar(np.arange(30), correct)
axs[1].bar(np.arange(30), incorrect, bottom=correct)


binary_label_modifier = {
    0: 0, # "Ball": "No hit",
    1: 0, # "Strike": "No hit",
    2: 1, # "Foul": "Batting",
    3: 0, # "Swing and a miss": "No hit",
    4: 1, # "Fly out": "Batting",
    5: 1, # "Ground out": "Batting",
    6: 1, # "One-base hit": "Batting",
    7: 0, # "Strike out": "No hit",
    8: 1, # "Home in": "Batting",
    9: 0, # "Base on balls": "No hit",
    10: 1, # "Touch out": "Batting",
    11: 1, # "Two-base hit": "Batting",
    12: 1, # "Homerun": "Batting",
    13: 1, # "Foul fly out": "Batting",
    14: 1, # "Double play": "Batting",
    15: 1, # "Tag out": "Batting",
    16: None, # "Stealing base": None,
    17: 1, # "Infield hit": "Batting",
    18: 1, # "Line-drive out": "Batting",
    19: 1, # "Error": "Batting",
    20: 0, # "Hit by pitch": "No hit",
    21: 1, # "Bunt foul": "Batting",
    22: 0, # "Wild pitch": "No hit",
    23: 1, # "Sacrifice bunt out": "Batting",
    24: None, # "Caught stealing": None,
    25: 1, # "Three-base hit": "Batting",
    26: 1, # "Bunt hit": "Batting",
    27: 1, # "Bunt out": "Batting",
    28: 0, # "Passed ball": "No hit",
    29: None, # "Pickoff out": None,
}

binary_labels = [0] * 30
for i in range(30):
    if binary_label_modifier[i] is None:
        binary_labels[i] = "None"
    elif binary_label_modifier[i] == 1:
        binary_labels[i] = "Batting"
    else:
        binary_labels[i] = "No hit"

plt.xticks(range(30))
axs[1].set_xticklabels(["{}({})".format(LABEL_ID_TO_STR[i], binary_labels[i]) for i in range(30)], rotation = 45, ha="right")
fig.subplots_adjust(bottom=0.2)

plt.show()
