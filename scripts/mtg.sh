# mkdir ./helper
cd ./helper
# git clone https://github.com/MTG/mtg-jamendo-dataset.git
cd mtg-jamendo-dataset
# pip install -r scripts/requirements.txt
python scripts/download/download.py --dataset raw_30s --type audio --from gdrive --unpack ../../../dataset/mtg