ARABIDOPSIS_ZIP_FILE=./datasets/arabidopsis_validation_samples.zip
BARLEY_ZIP_FILE=./datasets/barley_validation_samples.zip
# Maybe download araboidopsis images
if [ -f "$ARABIDOPSIS_ZIP_FILE" ]; then
    echo "$ARABIDOPSIS_ZIP_FILE exists"
else 
    echo "Downloading arabidopsis validation images..."
    curl https://cloudstor.aarnet.edu.au/plus/s/u3vy3YsGemQKgIJ/download \
        -o datasets/arabidopsis_validation_samples.zip
fi
# Maybe download barley images
if [ -f "$BARLEY_ZIP_FILE" ]; then
    echo "$BARLEY_ZIP_FILE exists"
else 
    echo "Downloading barley validation images..."
    curl https://cloudstor.aarnet.edu.au/plus/s/phlJcZkgS9y0fZd/download \
        -o datasets/barley_validation_samples.zip
fi

echo "Unzipping files..."
unzip -n $ARABIDOPSIS_ZIP_FILE -d ./datasets/arabidopsis/
unzip -n $BARLEY_ZIP_FILE -d ./datasets/barley/

ARABIDOPSIS_WEIGHTS_FILE=./arabidopsis_weights.pth
BARLEY_WEIGHTS_FILE=./barley_weights.pth
# Maybe download arabidopsis weights
if [ -f "$ARABIDOPSIS_WEIGHTS_FILE" ]; then
    echo "$ARABIDOPSIS_WEIGHTS_FILE exists"
else 
    echo "Downloading arabidopsis model weights..."
    curl https://cloudstor.aarnet.edu.au/plus/s/iLB4PwuKqjbdSWg/download \
        -o ./arabidopsis_weights.pth
fi
# Maybe download barley weights
if [ -f "$BARLEY_WEIGHTS_FILE" ]; then
    echo "$BARLEY_WEIGHTS_FILE exists"
else 
    echo "Downloading barley model weights..."
    curl https://cloudstor.aarnet.edu.au/plus/s/KWFjWBLlE18n9M9/download \
        -o ./barley_weights.pth
fi
