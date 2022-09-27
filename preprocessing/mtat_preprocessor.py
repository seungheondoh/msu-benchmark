import os
from collections import Counter
import random
import json
import torch
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
from contextlib import contextmanager
from audio_utils import load_audio
from io_utils import _json_dump
from constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE, KEY_DICT, MTAT_TAG_INFO

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def mtat_resampler(path):
    src, _ = load_audio(
        path=os.path.join(DATASET,'mtat','audio', path),
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    if src.shape[-1] < DATA_LENGTH: # short case
        pad = np.zeros(DATA_LENGTH)
        pad[:src.shape[-1]] = src
        src = pad
    elif src.shape[-1] > DATA_LENGTH: # too long case
        src = src[:DATA_LENGTH]
    save_name = os.path.join(DATASET,'mtat','npy', path.replace(".mp3",".npy"))
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    np.save(save_name, src.astype(np.float32))
    
def get_track_split(mtat_path, train, valid, test):
    track_split = {
        "train_track": [i.split("\t")[0] for i in train],
        "valid_track": [i.split("\t")[0] for i in valid],
        "test_track": [i.split("\t")[0] for i in test],
    }
    _json_dump(os.path.join(mtat_path, "track_split.json"), track_split)

def get_tag_info(mtat_path, tags, annotation):
    track_tag_matrix = annotation.set_index("clip_id")
    tag_statistics = track_tag_matrix.sum().to_dict()
    mtat_tag_info = MTAT_TAG_INFO.copy()
    _json_dump(os.path.join(mtat_path, "mtat_tags.json"), tags)
    _json_dump(os.path.join(mtat_path, "mtat_tag_info.json"), mtat_tag_info)
    _json_dump(os.path.join(mtat_path, "mtat_tag_stats.json"), tag_statistics)

def MTAT_processor(mtat_path):
    metadata = pd.read_csv(os.path.join(mtat_path, "clip_info_final.csv"), '\t').set_index("mp3_path")
    annotation = pd.read_csv(os.path.join(mtat_path, "annotations_final.csv"), sep='\t').set_index("mp3_path")
    tags = list(np.load(os.path.join(mtat_path, "split", "tags.npy")))
    binarys = np.load(os.path.join(mtat_path, "split", "binary.npy"))
    train = list(np.load(os.path.join(mtat_path, "split", "train.npy")))
    valid = list(np.load(os.path.join(mtat_path, "split", "valid.npy")))
    test = list(np.load(os.path.join(mtat_path, "split", "test.npy")))
    get_tag_info(mtat_path, tags, annotation)
    get_track_split(mtat_path, train, valid, test)
    total = train + valid + test
    results, mp3_paths = {}, []
    for item in total:
        ix, mp3_path = item.split('\t')
        binary = list(binarys[int(ix)])
        top_tag = [tags[idx] for idx, i in enumerate(binary) if i]
        meta_item = metadata.loc[mp3_path]
        anno_item = annotation.loc[mp3_path]
        extra_tag = list(anno_item[anno_item == 1].index)
        mp3_paths.append(mp3_path)
        results[ix] = {
            "track_id": ix,
            "tag": top_tag,
            "extra_tag": extra_tag,
            "clip_id": int(meta_item['clip_id']),
            "title": str(meta_item['title']),
            "artist_name": str(meta_item['artist']),
            "release": str(meta_item['album']),
            "path": str(meta_item.name),
        }
    _json_dump(os.path.join(mtat_path, "annotation.json"), results)
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # pool.map(mtat_resampler, mp3_paths)
    print("finish mtat extraction", len(results))

