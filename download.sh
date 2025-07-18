# Download all zip files
echo "Downloading dataset files..."
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/fire.zip
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/pumpkin.zip
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/stairs.zip

echo "Extracting main archives..."
unzip chess.zip 
unzip fire.zip 
unzip heads.zip 
unzip office.zip 
unzip pumpkin.zip 
unzip redkitchen.zip 
unzip stairs.zip

cd chess
for file in seq-*.zip; do
  unzip "$file"
  rm "$file"
done

cd ../fire
for file in seq-*.zip; do
  unzip "$file"
  rm "$file"
done

cd ../heads
for file in seq-*.zip; do
  unzip "$file"
  rm "$file"
done

cd ../office
for file in seq-*.zip; do
  unzip "$file"
  rm "$file"
done

cd ../pumpkin
for file in seq-*.zip; do
  unzip "$file"
  rm "$file"
done

cd ../redkitchen
for file in seq-*.zip; do
  unzip "$file"
  rm "$file"
done

cd ../stairs
for file in seq-*.zip; do
  unzip "$file"
  rm "$file"
done
cd ..

mkdir downloads/
mv chess downloads/chess
mv fire downloads/fire
mv heads downloads/heads
mv office downloads/office
mv pumpkin downloads/pumpkin
mv redkitchen downloads/redkitchen
mv stairs downloads/stairs

mkdir data/
python3 6DoF-Camera-Pose-Estimation/src/prepare_data.py --p downloads --o data
mv downloads 6DoF-Camera-Pose-Estimation/downloads
mv data 6DoF-Camera-Pose-Estimation/data
echo "Dataset preparation complete."