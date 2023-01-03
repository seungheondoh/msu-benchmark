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
from audio_utils import load_audio
from io_utils import _json_dump
from constants import DATASET, DATA_LENGTH, STR_CH_FIRST, MUSIC_SAMPLE_RATE, KEY_DICT, FMA_TAG_INFO

NaN_to_emptylist = lambda d: d if isinstance(d, list) or isinstance(d, str) else []
flatten_list_of_list = lambda l: [item for sublist in l for item in sublist]

def fma_load(filepath):
    filename = os.path.basename(filepath)
    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)
    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])
        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)
        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])
        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')
        return tracks

def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

def fma_resampler(track_id):
    # root_path = os.path.join(DATASET,'fma','fma_large')
    root_path = os.path.join(DATASET,'fma','fma_small')
    audio_path = get_audio_path(root_path, track_id)
    fname = audio_path.split(root_path + "/")[1]
    try:
        src, _ = load_audio(
            path=audio_path,
            ch_format= STR_CH_FIRST,
            sample_rate= MUSIC_SAMPLE_RATE,
            downmix_to_mono= True)
        if src.shape[-1] < DATA_LENGTH: # short case
            pad = np.zeros(DATA_LENGTH)
            pad[:src.shape[-1]] = src
            src = pad
        elif src.shape[-1] > DATA_LENGTH: # too long case
            src = src[:DATA_LENGTH]
        save_name = os.path.join(DATASET,'fma','npy', fname.replace(".mp3",".npy"))
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        np.save(save_name, src.astype(np.float32))
    except:
        save_name = os.path.join(DATASET,'fma','error', fname.replace(".mp3",".npy"))
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        np.save(save_name, track_id)
    
def get_track_split(tracks, target):
    target_subset = tracks['set', 'subset'] <= target
    train = tracks['set', 'split'] == 'training'
    val = tracks['set', 'split'] == 'validation'
    test = tracks['set', 'split'] == 'test'
    y_train = tracks.loc[target_subset & train, ('track', 'genre_top')]
    y_val = tracks.loc[target_subset & val, ('track', 'genre_top')]
    y_test = tracks.loc[target_subset & test, ('track', 'genre_top')]
    df_genre_top = pd.concat([y_train, y_val, y_test])
    traget_track_split = {
        "train_track": list(y_train.index),
        "valid_track": list(y_val.index),
        "test_track": list(y_test.index),
    }
    return traget_track_split, df_genre_top

def get_annotation(tracks, df_genre_top, fma_path):
    track_target = ['genres_all', 'genre_top', 'title']
    album_target = ['date_released','title']
    artist_target = ['name']
    df_track = tracks['track'][track_target]
    df_album = tracks['album'][album_target]
    df_artist = tracks['artist'][artist_target]
    df_album = df_album.rename(columns={"title": "release"})
    track_album = pd.merge(df_track, df_album, how='outer',on='track_id')
    df_fma = pd.merge(track_album, df_artist, how='outer',on='track_id')
    df_genre_top =  df_genre_top.map(lambda x: x.lower())
    annotation_dict = {}
    for track_id in df_genre_top.index:
        try:
            annotation_dict[track_id] = {
                "tag": str(df_genre_top.loc[track_id]),
                "title": str(df_fma.loc[track_id]['title']),
                "artist_name": str(df_fma.loc[track_id]['name']),
                "release": str(df_fma.loc[track_id]['release']),
                "year": str(df_fma.loc[track_id]['date_released'].year),
                "track_id": track_id,
            }
        except:
            annotation_dict[track_id] = {
                "tag": str(df_genre_top.loc[track_id]),
                "title": "",
                "artist_name": "",
                "release": "",
                "year": "",
                "track_id": track_id,
            }
    print(len(annotation_dict))
    _json_dump(os.path.join(fma_path, "annotation.json"), annotation_dict)
    df_annotation = pd.DataFrame(annotation_dict).T
    return annotation_dict, df_annotation

def get_tag_info(df_annotation, fma_path):
    fma_tag_info = FMA_TAG_INFO.copy()
    tags = list(set(df_annotation['tag']))
    tag_statistics = {i:j for i,j in Counter(tags).most_common()}
    _json_dump(os.path.join(fma_path, "fma_tags.json"), tags)
    _json_dump(os.path.join(fma_path, "fma_tag_info.json"), fma_tag_info)
    _json_dump(os.path.join(fma_path, "fma_tag_stats.json"), tag_statistics)

def FMA_processor(fma_path):
    tracks = fma_load(os.path.join(fma_path,"fma_metadata/tracks.csv"))
    genres = fma_load(os.path.join(fma_path,"fma_metadata/genres.csv"))
    small_track_split, df_genre_top = get_track_split(tracks, "small")
    total_track = small_track_split['train_track'] + small_track_split['valid_track']+ small_track_split['test_track']
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(fma_resampler, total_track)
    error_samples = []
    error_dir = os.path.join(DATASET,'fma','error')
    for dirs in os.listdir(error_dir):
        error_samples.extend(os.listdir(os.path.join(error_dir, dirs)))
    error_fnames = [int(i.split(".npy")[0]) for i in error_samples]
    tracks = tracks.drop(error_fnames, axis=0)
    annotation_dict, df_filter = get_annotation(tracks, df_genre_top, fma_path)
    get_tag_info(df_filter, fma_path)
    filtered_small = {
        "train_track": [track_id for track_id in small_track_split['train_track'] if track_id in annotation_dict.keys()],
        "valid_track": [track_id for track_id in small_track_split['valid_track'] if track_id in annotation_dict.keys()],
        "test_track": [track_id for track_id in small_track_split['test_track'] if track_id in annotation_dict.keys()]
    }
    _json_dump(os.path.join(fma_path, "track_split.json"), filtered_small)
    print("finish fma extraction", len(filtered_small['train_track']) + len(filtered_small['valid_track']) + len(filtered_small['test_track']))

