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
from constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE, KEY_DICT

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def gtzan_resampler(genre, path):
    src, _ = load_audio(
        path=os.path.join(DATASET,'gtzan','audio', genre, path),
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    if src.shape[-1] < DATA_LENGTH: # short case
        pad = np.zeros(DATA_LENGTH)
        pad[:src.shape[-1]] = src
        src = pad
    elif src.shape[-1] > DATA_LENGTH: # too long case
        src = src[:DATA_LENGTH]
    save_name = os.path.join(DATASET,'gtzan','npy', path.replace(".wav",".npy"))
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    np.save(save_name, src.astype(np.float32))
    
def get_track_split(gtzan_path):
    tr_names = os.path.join(gtzan_path, "GTZAN_split/train_filtered.txt")
    va_names = os.path.join(gtzan_path, "GTZAN_split/valid_filtered.txt")
    te_names = os.path.join(gtzan_path, "GTZAN_split/test_filtered.txt")
    track_split = {
        "train_track": open(tr_names,'r').read().splitlines(),
        "valid_track": open(va_names,'r').read().splitlines(),
        "test_track": open(te_names,'r').read().splitlines()
    }
    with open(os.path.join(gtzan_path, "track_split.json"), mode="w") as io:
        json.dump(track_split, io, indent=4)
    track_list = track_split['train_track'] + track_split['valid_track'] + track_split['test_track']
    track_list = [i.split("/")[1] for i in track_list]
    return track_list

def get_quantize_tempo(gtzan_path, track_list):
    df = pd.read_csv(os.path.join(gtzan_path, "GTZAN-Rhythm_v2_ismir2015_lbd/stats.csv"), index_col=0)
    gtzan_filtered = df.loc[track_list]
    gtzan_filtered = gtzan_filtered.rename(columns = {'artist' : 'artist_name'})
    gtzan_filtered['tempo'] = gtzan_filtered['tempo mean'].astype(int)
    gtzan_tempo = gtzan_filtered[["artist_name","title","tempo"]]
    return gtzan_tempo

def get_key_info(gtzan_path, gtzan_tempo):
    root = os.path.join(gtzan_path, "gtzan_key/gtzan_key/genres/")
    tag_list, key_list = [], []
    for i in gtzan_tempo.index:
        genre = i.split(".")[0]
        fname = i.replace('.wav','.lerch.txt')
        value = open(os.path.join(root, genre, fname),'r').read()
        key_list.append(KEY_DICT[value.strip()])
        tag_list.append(genre)
    gtzan_tempo['key'] = key_list
    gtzan_tempo['tag'] = tag_list
    return gtzan_tempo

def GTZAN_processor(gtzan_path):
    track_list = get_track_split(gtzan_path)
    gtzan_tempo = get_quantize_tempo(gtzan_path, track_list)
    gtzan_tempo_key = get_key_info(gtzan_path, gtzan_tempo)
    gtzan_final = gtzan_tempo_key[["artist_name","title","key","tempo","tag"]]
    gtzan_final.index.name = "track_id"
    gtzan_final['track_id'] = gtzan_final.index
    gtzan_final = gtzan_final.fillna(0)
    kv_dataset = gtzan_final.to_dict('index')
    tag_stats = {i:j for i,j in Counter(gtzan_final['tag']).most_common()}
    tag_info = {i:["genre"] for i in set(gtzan_final['tag'])}
    _json_dump(os.path.join(gtzan_path, 'annotation.json'), kv_dataset)
    _json_dump(os.path.join(gtzan_path, 'gtzan_tags.json'), list(set(gtzan_final['tag'])))
    _json_dump(os.path.join(gtzan_path, 'gtzan_tag_stats.json'), tag_stats)
    _json_dump(os.path.join(gtzan_path, 'gtzan_tag_info.json'), tag_info)
    
    with poolcontext(processes=multiprocessing.cpu_count()-5) as pool:
        pool.starmap(gtzan_resampler, zip(list(gtzan_final['tag']), list(gtzan_final.index)))

    print("finish gtzan extract", len(track_list))