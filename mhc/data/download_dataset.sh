# cd ase/data
# bash download_dataset.sh

echo -e "Downloading amass_processed_HumanML3D, the dataset processed for smpl_humanoid.xml"
gdown --fuzzy https://drive.google.com/file/d/1J1ePaP1oZNEArilYs7hKgvQ1TeyYHQcY/view?usp=sharing

echo -e "unzipping amass_processed_HumanML3D.zip"
unzip amass_processed_HumanML3D.zip

echo -e "Cleaning\n"
rm -rf amass_processed_HumanML3D.zip

echo -e "All done!"