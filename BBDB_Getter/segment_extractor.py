import os
import json
from moviepy.editor import VideoFileClip

bbdb_path = './BBDB/fullgame'
segments_path = './segments'
def extract(filename, meta):
    gamecode, _ = os.path.splitext(filename)
    clips_path = os.path.join(segments_path, gamecode)
    if not os.path.exists(clips_path):
        os.mkdir(clips_path)
    clip = VideoFileClip(os.path.join(bbdb_path, filename))
    for i, anno in enumerate(meta['database'][gamecode]['annotations']):
        new_filename = "{}.mp4".format(i)
        new_filepath = os.path.join(clips_path, new_filename)
        if os.path.isfile(new_filepath):
            continue
        subclip = clip.subclip(anno['segment'][0], anno['segment'][1])
        subclip.write_videofile(new_filepath)


def main():
    if not os.path.exists(segments_path):
        os.mkdir(segments_path)
    with open('./bbdb.v0.9.min.json', 'r') as fp:
        meta = json.load(fp)
    for _, _, files in os.walk(bbdb_path):
        for file in files:
            if '.mp4' in file:
                extract(file, meta)

if __name__ == "__main__":
  main()