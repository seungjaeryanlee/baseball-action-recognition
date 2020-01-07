import os
import skvideo.io
import yaml
import json


meta_path = './bbdb.v0.9.min.json'
labels_path = './labels.yaml'
segments_path = './segments'


def labels():
    with  open(labels_path) as fp:
        ls = yaml.load(fp, Loader=yaml.FullLoader)
    return ls


class Video:
    def __init__(self, gamecode, segment_index, fps, label_index):
        self.gamecode = gamecode
        self.segment_index = segment_index
        self.fps = fps
        self.label_index = label_index
    
    def data(self, skip_rate):
        if self.fps > 30:
            skip_rate *= 2
        filename = "{}.mp4".format(self.segment_index)
        filepath = os.path.join(segments_path, self.gamecode, filename)
        videodata = skvideo.io.vread(filepath)
        return videodata[::skip_rate]


def get_videos(gamecodes=None, label_indexes=None):
    with open(meta_path) as fp:
        meta = json.load(fp)
    dirs = os.listdir(segments_path)
    if gamecodes is not None:
        dirs = [d for d in dirs if d in gamecodes]
    videos = []
    for gamecode in dirs:
        fps = meta["database"][gamecode]["fps"]
        for i, segment in enumerate(meta["database"][gamecode]["annotations"]):
            label_index = segment["labelIndex"]
            if label_indexes is not None and segment and\
                label_index not in label_indexes:
                continue
            videos.append(Video(gamecode, i, fps, label_index))
    return videos


def extract_data(videos, skip_rate):
    vlist = [video.data(skip_rate) for video in videos]
    vanno = [video.label_index for video in videos]
    return vlist, vanno
