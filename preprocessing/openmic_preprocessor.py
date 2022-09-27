import os
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
from constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE, KEY_DICT

NaN_to_emptylist = lambda d: d if isinstance(d, list) or isinstance(d, str) else []
flatten_list_of_list = lambda l: [item for sublist in l for item in sublist]

def openmic_resampler(sample_key):
    fname = id_to_audio_filename(sample_key)
    audio_path = os.path.join(DATASET, 'openmic', 'openmic-2018', fname)
    src, _ = load_audio(
        path=audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    if src.shape[-1] > DATA_LENGTH: # too long case
        src = src[:DATA_LENGTH]
    save_name = os.path.join(DATASET,'openmic','npy', sample_key + ".npy")
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    np.save(save_name, src.astype(np.float32))

def id_to_audio_filename(sample_key: str, ext='ogg'):
    return os.path.join('audio', sample_key[:3], '%s.%s' % (sample_key, ext))

def _id_to_target_vector(_id, n_inst, df_label, inst_to_idx):
    target = np.zeros([n_inst,], dtype=np.float32)
    inst_idxs = []
    for label in df_label[df_label.sample_key == _id].instrument.to_numpy():
        inst_idx = inst_to_idx[label]
        inst_idxs.append(inst_idx)
        target[inst_idx] = 1.0
    return target, inst_idxs

def get_track_split(openmic_path, df_train, df_test, n_inst, df_label, inst_to_idx):
    X, Y = [],[]
    for _id in list(df_train[0]):
        binary, category = _id_to_target_vector(_id, n_inst, df_label, inst_to_idx)
        X.append(_id)
        Y.append(binary)
    new_X = np.array([(idx, x) for idx, x in enumerate(X)])
    trX,trY,vaX,vaY= iterative_train_test_split(new_X, np.stack(Y), test_size = 0.1)
    track_split = {
        "train_track": list(trX[:,-1:].squeeze(-1)),
        "valid_track": list(vaX[:,-1:].squeeze(-1)),
        "test_track": list(df_test[0])
    }
    _json_dump(os.path.join(openmic_path, f"track_split.json"), track_split)
    return track_split

def get_tag_info(insts, df_label, openmic_path):
    all_tags = insts
    openmic_tag_info = {inst:"instrument" for inst in insts}
    tag_statistics = {i:j for i,j in Counter(list(df_label['instrument'])).most_common()}
    _json_dump(os.path.join(openmic_path, f"openmic_tags.json"), list(set(all_tags)))
    _json_dump(os.path.join(openmic_path, f"openmic_tag_info.json"), openmic_tag_info)
    _json_dump(os.path.join(openmic_path, f"openmic_tag_stats.json"), tag_statistics)
    
def isNaN(num):
    return num != num

def get_annotation(df_meta, df_label, openmic_path):
    target_col = ["track_id", "sample_key", "artist_name", "track_title", "track_date_created", "track_genres"]
    df_target = df_meta[target_col]
    df_label = df_label.set_index("sample_key")
    results = {}
    for idx in range(len(df_target)):
        item = df_target.iloc[idx]
        tag = df_label.loc[item['sample_key']]['instrument']
        if type(tag) == str:
            tag = [tag]
        else:
            tag = list(tag)
        results[item['sample_key']] = {
            "tag": tag,
            "title": item['track_title'],
            "artist_name": item['artist_name'],
            "year": item['track_date_created'].split(" ")[0].split("/")[-1],
            "track_id": item['sample_key']
        }
    _json_dump(os.path.join(openmic_path, "annotation.json"), results)
    return pd.DataFrame(results).T

def OPENMIC_processor(openmic_path):
    df_train = pd.read_csv(os.path.join(openmic_path, "openmic-2018/partitions/split01_train.csv"), header=None)
    df_test = pd.read_csv(os.path.join(openmic_path, "openmic-2018/partitions/split01_test.csv"), header=None)
    df_meta = pd.read_csv(os.path.join(openmic_path, "openmic-2018/openmic-2018-metadata.csv"))
    df_label = pd.read_csv(os.path.join(openmic_path, "openmic-2018/openmic-2018-aggregated-labels.csv"), header=0)
    df_id = list(df_train[0]) + list(df_test[0])
    insts = sorted(df_label.instrument.unique().tolist())
    inst_to_idx = {val: key for key, val in enumerate(insts)}
    n_inst = df_label.instrument.nunique()
    get_tag_info(insts, df_label, openmic_path)
    track_split = get_track_split(openmic_path, df_train, df_test, n_inst, df_label, inst_to_idx)
    df_annotation = get_annotation(df_meta, df_label, openmic_path)
    track_split = json.load(open(os.path.join(openmic_path, "track_split.json"), 'r'))
    total_track = track_split['train_track'] + track_split['valid_track'] + track_split['test_track']
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(openmic_resampler, total_track)
    print("finish openmic extract", len(total_track))

