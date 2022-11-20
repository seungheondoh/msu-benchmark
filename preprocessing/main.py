import os
from gtzan_preprocessor import GTZAN_processor
from mtat_preprocessor import MTAT_processor
from fma_preprocessor import FMA_processor
from mtg_preprocessor import MTG_processor
from kvt_preprocessor import KVT_processor
from emotify_preprocessor import EMOTIFY_processor
from constants import DATASET

def main():
    GTZAN_processor(gtzan_path=os.path.join(DATASET, 'gtzan'))
    MTAT_processor(mtat_path=os.path.join(DATASET, 'mtat'), split_type="codified")
    FMA_processor(fma_path=os.path.join(DATASET, 'fma'))
    MTG_processor(mtg_path=os.path.join(DATASET, 'mtg'))
    KVT_processor(kvt_path=os.path.join(DATASET, 'kvt'))
    EMOTIFY_processor(emotify_path=os.path.join(DATASET, 'emotify'))

if __name__ == '__main__':
    main()