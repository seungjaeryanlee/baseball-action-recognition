# baseball-action-recognition

## Prerequisites

This repository was tested in Ubuntu 18.04 LTS with Python 3.7.4.

The Python requirements are written in `requirements.txt`. Additionally, to download the Baseball Database (BBDB), you need `aria2`.

## Prepare the Dataset

This repository has two script files to

1. Download full videos from BBDB, and
2. Extract segments from full videos

Run the two lines below in order

```bash
python download_bbdb.py -i bbdb.selected.v0.9.min.json
python extract_segments_from_videos.py
```
