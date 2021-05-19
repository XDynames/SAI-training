ARABIDOPSIS_ZIP_FILE=./datasets/arabidopsis_validation_samples.zip
BARLEY_ZIP_FILE=./datasets/barley_validation_samples.zip
# Maybe download araboidopsis images
if [ -f "$ARABIDOPSIS_ZIP_FILE" ]; then
    echo "$ARABIDOPSIS_ZIP_FILE exists"
else 
    echo "Downloading arabidopsis validation images..."
    curl https://cloudstor.aarnet.edu.au/plus/s/u3vy3YsGemQKgIJ/download \
        -o $ARABIDOPSIS_ZIP_FILE
fi
# Maybe download barley images
if [ -f "$BARLEY_ZIP_FILE" ]; then
    echo "$BARLEY_ZIP_FILE exists"
else 
    echo "Downloading barley validation images..."
    curl https://cloudstor.aarnet.edu.au/plus/s/phlJcZkgS9y0fZd/download \
        -o $BARLEY_ZIP_FILE
fi

mkdir -p ./datasets/arabidopsis/stoma/
mkdir -p ./datasets/barley/stoma/

echo "Unzipping files..."
unzip -n $ARABIDOPSIS_ZIP_FILE -d ./datasets/arabidopsis/stoma/
unzip -n $BARLEY_ZIP_FILE -d ./datasets/barley/stoma/

mkdir -p ./datasets/arabidopsis/stoma/annotations/
mkdir -p ./datasets/barley/stoma/annotations/

ARABIDOPSIS_ANNOTATIONS_FILE=./datasets/arabidopsis/stoma/annotations/val.json
BARELY_ANNOTATIONS_FILE=./datasets/barley/stoma/annotations/val.json
# Maybe download ground truth for barley
if [ -f "$BARELY_ANNOTATIONS_FILE" ]; then
    echo "$BARELY_ANNOTATIONS_FILE exists"
else  
    echo "Downloading barley annotations..."
    curl https://cloudstor.aarnet.edu.au/plus/s/7kQNLKVwcJM7kcB/download \
        -o $BARELY_ANNOTATIONS_FILE
fi
# Maybe download ground truth for arabidopsis
if [ -f "$ARABIDOPSIS_ANNOTATIONS_FILE" ]; then
    echo "$ARABIDOPSIS_ANNOTATIONS_FILE exists"
else 
    echo "Downloading arabidopsis annotations..."
    curl https://cloudstor.aarnet.edu.au/plus/s/TIDaxt8AxajeruY/download \
        -o $ARABIDOPSIS_ANNOTATIONS_FILE
fi


ARABIDOPSIS_WEIGHTS_FILE=./arabidopsis_weights.pth
BARLEY_WEIGHTS_FILE=./barley_weights.pth
COMBINED_WEIGHTS_FILE=./combined_weights.pth
# Maybe download arabidopsis weights
if [ -f "$ARABIDOPSIS_WEIGHTS_FILE" ]; then
    echo "$ARABIDOPSIS_WEIGHTS_FILE exists"
else 
    echo "Downloading arabidopsis model weights..."
    curl https://cloudstor.aarnet.edu.au/plus/s/iLB4PwuKqjbdSWg/download \
        -o $ARABIDOPSIS_WEIGHTS_FILE
fi
# Maybe download barley weights
if [ -f "$BARLEY_WEIGHTS_FILE" ]; then
    echo "$BARLEY_WEIGHTS_FILE exists"
else 
    echo "Downloading barley model weights..."
    curl https://cloudstor.aarnet.edu.au/plus/s/KWFjWBLlE18n9M9/download \
        -o $BARLEY_WEIGHTS_FILE
fi
# Maybe download combined weights
if [ -f "$COMBINED_WEIGHTS_FILE" ]; then
    echo "$COMBINED_WEIGHTS_FILE exists"
else 
    echo "Downloading combined model weights..."
    curl https://cloudstor.aarnet.edu.au/plus/s/EQMljoS9YLvpHtS/download \
        -o $COMBINED_WEIGHTS_FILE
fi
echo "Installing SAI..."
python setup.py build develop