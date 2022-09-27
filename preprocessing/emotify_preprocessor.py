import os
from collections import Counter
import json
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
from contextlib import contextmanager
from audio_utils import load_audio
from io_utils import _json_dump
from sklearn.model_selection import train_test_split
from constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE, KEY_DICT

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def emotify_resampler(path):
    src, _ = load_audio(
        path=os.path.join(DATASET,'emotify','audio',path),
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    if src.shape[-1] < DATA_LENGTH: # short case
        pad = np.zeros(DATA_LENGTH)
        pad[:src.shape[-1]] = src
        src = pad
    elif src.shape[-1] > DATA_LENGTH: # too long case
        src = src[:DATA_LENGTH]
    save_name = os.path.join(DATASET,'emotify','npy', path.replace(".mp3",".npy"))
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    np.save(save_name, src.astype(np.float32))

def get_annotation(df, emotify_path):
    target_emotion = ["amazement","solemnity","tenderness","nostalgia","calmness","power","joyful_activation","tension","sadness"]
    annotation = {}
    for idx in range(1,401):
        item = df[df['track id'] == idx]
        track_id = str(idx)
        parsing_id = int(track_id[:len(str(idx))])
        if len(str(idx)) == 3:
            parsing_id = int(track_id[1:len(str(idx))])
            if idx % 10 == 0:
                parsing_id = str(100)
        genre = list(item['genre'])[0]
        num_of_annotator = len(item)
        score = item[target_emotion].sum()
        score_ratio = score / num_of_annotator
        tag = score_ratio.idxmax()
        annotation[str(idx)] = {
            "track_id": str(idx),
            "path": f"{genre}/{parsing_id}.mp3",
            "genre": genre,
            "num_of_annotator": str(num_of_annotator),
            "score": [int(i) for i in score.values],
            "tag_list": list(score.index),
            "tag": tag
        }
    df_annotation = pd.DataFrame(annotation).T
    tag_stats = {i:j for i,j in Counter(df_annotation['tag']).most_common()}
    tag_info = {i:["genre"] for i in set(df_annotation['tag'])}
    _json_dump(os.path.join(emotify_path, "annotation.json"), annotation)
    _json_dump(os.path.join(emotify_path, 'emotify_tags.json'), target_emotion)
    _json_dump(os.path.join(emotify_path, 'emotify_tag_stats.json'), tag_stats)
    _json_dump(os.path.join(emotify_path, 'emotify_tag_info.json'), tag_info)
    return df_annotation

def get_track_split(df_annotation, emotify_path):
    trs,vas,tes = [], [], []
    for tag in set(df_annotation['tag']):
        if tag == "amazement":
            pool = df_annotation[df_annotation['tag'] == tag]
            trX = pd.DataFrame(pool.iloc[0]).T
            vaX = pd.DataFrame(pool.iloc[1]).T
            teX = pd.DataFrame(pool.iloc[2]).T
        else:
            pool = df_annotation[df_annotation['tag'] == tag]
            try:
                trvaX, teX= train_test_split(pool, stratify=pool['genre'], test_size = 0.187, random_state=42)
                trX, vaX = train_test_split(trvaX, stratify=trvaX['genre'], test_size = 0.113, random_state=42)
            except:
                trvaX, teX= train_test_split(pool, test_size = 0.187, random_state=42)
                trX, vaX = train_test_split(trvaX, test_size = 0.113, random_state=42)
        trs.append(trX)
        vas.append(vaX)
        tes.append(teX)
    track_split = {
        "train_track":list(pd.concat(trs)['track_id'].sort_values()),
        "valid_track": list(pd.concat(vas)['track_id'].sort_values()),
        "test_track": list(pd.concat(tes)['track_id'].sort_values())
    }
    _json_dump(os.path.join(emotify_path, "track_split.json"), track_split)


def EMOTIFY_processor(emotify_path):
    df = pd.read_csv(os.path.join(emotify_path, "data.csv"))
    df = df.rename(columns={i:i.strip() for i in df.columns})
    df_annotation = get_annotation(df, emotify_path)
    get_track_split(df_annotation, emotify_path)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(emotify_resampler, list(df_annotation['path']))
    print("finish emotify extract", len(df_annotation))