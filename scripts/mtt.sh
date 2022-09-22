cd ../dataset/mtt/
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003
cat mp3.zip.* > mp3_all.zip
unzip mp3_all.zip

rm -rf mp3_all.zip
rm -rf mp3.zip.*