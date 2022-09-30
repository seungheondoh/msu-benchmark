import os
import pickle
from collections import Counter
import random
import ast
import json
import torch
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
from contextlib import contextmanager
from skmultilearn.model_selection import iterative_train_test_split
from audio_utils import load_audio
from io_utils import _json_dump
from constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE, KEY_DICT, KVT_ARTIST

NaN_to_emptylist = lambda d: d if isinstance(d, list) or isinstance(d, str) else []
flatten_list_of_list = lambda l: [item for sublist in l for item in sublist]

def kvt_resampler(fname):
    audio_path = os.path.join(DATASET, 'kvt', 'audio', fname + ".mp3")
    src, _ = load_audio(
        path=audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    save_name = os.path.join(DATASET,'kvt','npy', fname + ".npy")
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    np.save(save_name, src.astype(np.float32))


def get_tag_info(tags, df_kvt, kvt_path):
    all_tags = [tag.lower() for tag in tags]
    kvt_tag_info = {i:"vocal" for i in all_tags}
    tag_statistics = {i.lower():j for i,j in Counter(flatten_list_of_list(df_kvt['tag'])).most_common()}
    _json_dump(os.path.join(kvt_path, "kvt_tags.json"), all_tags)
    _json_dump(os.path.join(kvt_path, "kvt_tag_info.json"), kvt_tag_info)
    _json_dump(os.path.join(kvt_path, "kvt_tag_stats.json"), tag_statistics)

def get_annotation(X,Y, df_meta, kvt_path, artists):
    translate_map = {ko:en.strip() for en,ko in zip(KVT_ARTIST.split(","), artists)}
    annotation = {}
    for x,y in zip(X,Y):
        _id = x.replace(".mp3","")
        meta = df_meta.loc[x]
        binary = []
        for tag, value in y.items():
            if value > 1.0:
                binary.append(1)
            else:
                binary.append(0)
        if np.array(binary).sum() > 1:
            annotation[_id] = {
                "track_id": _id,
                "tag": [tag.strip().lower() for tag, value in y.items() if value > 1.0],
                "artist": translate_map[meta['artist']].lower(),
                "title": meta['title']
            }
    _json_dump(os.path.join(kvt_path, "annotation.json"), annotation)
    df_kvt = pd.DataFrame(annotation).T
    return df_kvt, annotation

def KVT_processor(kvt_path):
    split_segment = pickle.load(open(os.path.join(kvt_path, "kpop_split", "split_segment.pkl"), 'rb'))
    df_meta = pd.DataFrame(split_segment['train'] + split_segment['valid'] + split_segment['test'])
    df_meta = df_meta.set_index("fileName")
    train_labels = pickle.load(open(os.path.join(kvt_path, "kpop_split", "train_labels.pkl"), 'rb'))
    train_files = pickle.load(open(os.path.join(kvt_path, "kpop_split", "train_files.pkl"), 'rb'))
    valid_labels = pickle.load(open(os.path.join(kvt_path,"kpop_split",  "valid_labels.pkl"), 'rb'))
    valid_files = pickle.load(open(os.path.join(kvt_path, "kpop_split", "valid_files.pkl"), 'rb'))
    test_labels = pickle.load(open(os.path.join(kvt_path, "kpop_split", "test_labels.pkl"), 'rb'))
    test_files = pickle.load(open(os.path.join(kvt_path, "kpop_split", "test_files.pkl"), 'rb'))
    artists = pickle.load(open(os.path.join(kvt_path, "artists.pkl"), 'rb'))
    X = train_files + valid_files + test_files
    Y = train_labels + valid_labels + test_labels
    df_kvt, annotation = get_annotation(X,Y, df_meta, kvt_path, artists)
    tags = pickle.load(open(os.path.join(kvt_path, "tags.pkl"), 'rb'))
    get_tag_info(tags, df_kvt, kvt_path)
    track_split = {
        "train_track": [i.replace(".mp3","") for i in train_files if i.replace(".mp3","") in annotation.keys()],
        "valid_track": [i.replace(".mp3","") for i in valid_files if i.replace(".mp3","") in annotation.keys()],
        "test_track": [i.replace(".mp3","") for i in test_files if i.replace(".mp3","") in annotation.keys()]
    }
    _json_dump(os.path.join(kvt_path, "track_split.json"), track_split)
    total_track = track_split['train_track'] + track_split['valid_track'] + track_split['test_track']
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-4)
    pool.map(kvt_resampler, total_track)
    print("finish kvt extract", len(total_track))