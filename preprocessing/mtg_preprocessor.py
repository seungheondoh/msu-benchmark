import os
import csv
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
from constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE, KEY_DICT, JAMENDO_TAG_INFO

NaN_to_emptylist = lambda d: d if isinstance(d, list) or isinstance(d, str) else []
flatten_list_of_list = lambda l: [item for sublist in l for item in sublist]


def jamendo_resampler(track_id):
    audio_path = os.path.join(DATASET, 'mtg', 'raw30s', track_id)
    src, _ = load_audio(
        path=audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    save_name = os.path.join(DATASET,'mtg','npy', track_id.replace(".mp3",".npy"))
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    np.save(save_name, src.astype(np.float32))

def read_file(tsv_file):
    tracks = {}
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                'path': row[3],
                'tag': row[5:],
            }
    return tracks

def get_split(mtg_path, split_type):
    train = read_file(os.path.join(mtg_path, "split-0", f"{split_type}-train.tsv"))
    validation = read_file(os.path.join(mtg_path, "split-0", f"{split_type}-validation.tsv"))
    test = read_file(os.path.join(mtg_path, "split-0", f"{split_type}-test.tsv"))
    track_split = {
        "train_track": list(train.keys()),
        "valid_track": list(validation.keys()),
        "test_track": list(test.keys())
    }
    _json_dump(os.path.join(mtg_path, f"{split_type}_track_split.json"), track_split)
    return track_split

def get_tag_split(df_total, mtg_path, split_type=False):
    if split_type:
        tags = [i.split("---")[1] for i in set(flatten_list_of_list(df_total['tag']))]
        _json_dump(os.path.join(mtg_path, f"mtg_{split_type}_tags.json"), tags)
    else:
        jamendo_tag_info = JAMENDO_TAG_INFO.copy()
        all_tags = flatten_list_of_list(list(df_total['tag']))
        tag_statistics= {i.split("---")[1]:j for i,j in Counter(all_tags).most_common()}
        tags = list(tag_statistics.keys())
        print("number of tag",len(tags))
        _json_dump(os.path.join(mtg_path, f"mtg_tags.json"), tags)
        _json_dump(os.path.join(mtg_path, f"mtg_tag_info.json"), jamendo_tag_info)
        _json_dump(os.path.join(mtg_path, f"mtg_tag_stats.json"), tag_statistics)


def get_annotation(mtg_path, split_type=False):
    if split_type:
        train = read_file(os.path.join(mtg_path, "split-0", f"{split_type}-train.tsv"))
        validation = read_file(os.path.join(mtg_path, "split-0", f"{split_type}-validation.tsv"))
        test = read_file(os.path.join(mtg_path, "split-0", f"{split_type}-test.tsv"))
    else: 
        train = read_file(os.path.join(mtg_path, "split-0", "autotagging-train.tsv"))
        validation = read_file(os.path.join(mtg_path, "split-0", "autotagging-validation.tsv"))
        test = read_file(os.path.join(mtg_path, "split-0", "autotagging-test.tsv"))
    total = {}
    total.update(train)
    total.update(validation)
    total.update(test)
    annotation = {}
    for track_id, path_tags in total.items():
        annotation[track_id] = {
            "track_id": track_id,
            "path": path_tags['path'],
            "tag": [tag.split("---")[1] for tag in path_tags['tag']]
        }
    if split_type:
        _json_dump(os.path.join(mtg_path, f"{split_type}_annotation.json"), annotation)
    else:
        _json_dump(os.path.join(mtg_path, "annotation.json"), annotation)
    return pd.DataFrame(total).T
    

def MTG_processor(mtg_path):
    for split_type in ['autotagging_top50tags', 'autotagging_genre','autotagging_moodtheme','autotagging_instrument']:
        split_info = get_split(mtg_path, split_type=split_type)
        df_annotation = get_annotation(mtg_path, split_type=split_type)
        get_tag_split(df_annotation, mtg_path, split_type=split_type)
    track_split = get_split(mtg_path, split_type="autotagging")
    df_total = get_annotation(mtg_path)
    get_tag_split(df_total, mtg_path)
    mp3_path = list(df_total['path'])
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # pool.map(jamendo_resampler, mp3_path)
    print("finish jamendo extract", len(df_total))