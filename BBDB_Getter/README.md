# BBDB_Getter
## How to use
### Download dependencies

```
pip install -r requirement.txt
```

### Download video files
Download raw video files of BBDB.

```
pytho bbdb_downloader.py -n 4
```

#### options
- `-n` : Specify the number of videos you install

**caution** Each file contains 1.5 ~ 4.0 GB of data

### Split video files into segments
Split the downloaded videos into the annotated segments.

```
python segment_extractor.py
```

### Get data from the annotated segments specifying how many frames you skip

```python
import video_getter
labels = video_getter.labels()
labels
# => ['Ball', 'Strike', 'Foul', 'Swing and a miss', 'Fly out', 'Ground out', 'One-base hit', 'Strike out', 'Home in', 'Base on balls', 'Touch out', 'Two-base hit', 'Homerun', 'Foul fly out', 'Double play', 'Tag out', 'Stealing base', 'Infield hit', 'Line-drive out', 'Error', 'Hit by pitch', 'Bunt foul', 'Wild pitch', 'Sacrifice bunt out', 'Caught stealing', 'Three-base hit', 'Bunt hit', 'Bunt out', 'Passed ball', 'Pickoff out']

#Extract 'Home in' and 'Homerun' scenes
label_indexes = [8, 12]

# collect metadata for videos of given queries. If no argument is given, the metadata of all existing segments are collected.
videos = video_getter.get_videos(label_indexes=label_indexes)

# Get video data. In this case, only one frame for each 10 frames is extracted.
video_data = videos[0].data(10)
video_data.shape
# => (45, 720, 1280, 3)
video_data = videos[0].data(30)
video_data.shape
# => (15, 720, 1280, 3)

labels[videos[0].label_index]
# => 'Home in'

# Get the data and labels for all videos. !caution! it require huge memory space.
vlist, vanno = video_getter.extract_data(videos[0:5], 150)
vlist[0].shape
# => (3, 720, 1280, 3)
vanno
# => [8, 12, 8, 8, 8]
```