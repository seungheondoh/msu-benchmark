# Music Semantic Understanding Benchmark


### Quick Start
```
bash scripts/download_splits.sh
```

### Dataset
The selection criteria are as follows: if a dataset has 1) commercial music for retrieval, 2) publicly assessed (at least upon request) and 3) categorical single or multi-label annotations for supporting text-based retrieval scenarios. 

| **Dataset** | # of Clip | # of Label | Avg.Tag | Task | Src |
|---|:---:|:---:|:---:|:---:|:---:|
| **MTAT1** | 25,860 | 50 | 2.70 | Tagging | [Link](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset), [Split](https://github.com/p-lambda/jukemir) |
| **MTAT2** | 21,108 | 50 | 3.30 | Tagging | [Link](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset), [Split](https://github.com/minzwon/sota-music-tagging-models) |
| **MTG top50s** | 54,380 | 50 | 3.07 | Tagging | [Link](https://github.com/MTG/mtg-jamendo-dataset) |
| **MTG Genre** | 55,094 | 87 | 2.44 | Genre | [Link](https://github.com/MTG/mtg-jamendo-dataset) |
| **FMA Small** | 8,000 | 8 | 1 | Genre | [Link](https://github.com/mdeff/fma) |
| **GTZAN** | 930 | 10 | 1 | Genre | [Link](http://opihi.cs.uvic.ca/sound/genres.tar.gz), [Split](https://github.com/jongpillee/music_dataset_split/tree/master/GTZAN_split) |
| **MTG Inst** | 24,976 | 40 | 2.57 | Instrument | [Link](https://github.com/MTG/mtg-jamendo-dataset) |
| **KVT** | 6,787 | 42 | 22.78 | Vocal | [Link](https://khlukekim.github.io/kvtdataset/) |
| **MTG Mood Theme** | 17,982 | 56 | 1.77 | Mood/Theme | [Link](https://github.com/MTG/mtg-jamendo-dataset) |
| **Emotify** | 400 | 9 | 1 | Mood | [Link](http://www2.projects.science.uu.nl/memotion/emotifydata/) |

We summarize all the datasets and tasks in Table. MagnaTagATune (MTAT) consists of 25k music clips from 5,223 unique songs. Following a previous work, we use their published splits and top~50 tags. We do not compare result with previous works using different split. MTG-Jamendo (MTG) contains 55,094 full audio tracks with 183 tags about genre, instrument, and mood/theme. We use the official splits (split-0) in each category for tagging, genre, instrument, and mood/theme tasks. For single-label genre classification, we use the fault-filtered version of GTZAN (GZ) and the `small' version of Free Music Archive (FMA-Small). For the vocal attribute recognition task, we use K-pop Vocal Tag (KVT) dataset. It consists of 6,787 vocal segments from K-pop music tracks. All the segments are annotated with 42 semantic tags describing various vocal style including pitch range, timbre, playing techniques, and gender. For the categorical mood recognition task, we use Emotify dataset. It consists of 400 excerpts in 4 genres with 9 emotional categories.

### Why we made this repo
There are now too many datasets and too many data splits. Because of this, if you are using multi-datasets, creating a loader will cost you a lot of time. To solve this, we propose a preprocessor for making `KV Style (key-values) annotation` file, `track split` file, and `resampler`. This will help the re-implementation of the research.


### Re-implementation
```
# download split and audio files
bash scripts/download.sh

# preprocessing all
cd preprocessing
python main.py
```

### Dataset Request
If you have difficulty accessing the dataset audio file, please contact- seungheondoh@kaist.ac.kr

### Reference
