import json
import glob


selected_gamecodes = glob.glob("./BBDB/fullgame/*.mp4")
selected_gamecodes = [ gamecode.replace("./BBDB/fullgame/", "").replace(".mp4", "") for gamecode in selected_gamecodes ]

with open("bbdb.v0.9.min.json", "r") as fp:
    data = json.load(fp)

selected_data = {
    'version': data['version'],
    'database': {k: v for k, v in data['database'].items() if k in selected_gamecodes },
}

with open("bbdb.selected.v0.9.min.json", "w+") as fp:
    json.dump(selected_data, fp)
