#!/bin/bash
#
# Runs ActionVLAD (RGB) model on a video
#
# Usage:
#    ./run.sh <VIDEO_PATH>

function usage() {
    echo "Usage: ./$0 <video_path>"
}

if [[ "$#" != 1 ]] ; then
    usage
    exit 1
fi

if [[ "$1" == "--help" || "$1" == "-h" ]] ; then
    usage
    exit 0
fi

vpath=$1
TMPDIR="/tmp/ActionVLAD-DEMO/"
echo "Using $TMPDIR for storing temporary data. WILL BE DELETED."
rm -r $TMPDIR
mkdir -p $TMPDIR
FRAME_DIR=$TMPDIR/frames/
LIST_DIR=$TMPDIR/lists/
FEAT_FILE=$TMPDIR/feats.h5
mkdir $FRAME_DIR
mkdir $LIST_DIR

# 1. extract frames, change q:v to qscale for older ffmpeg
ffmpeg -i $vpath -qscale:v 1 $TMPDIR/frames/image_%05d.jpg < /dev/null

# 2. Set up the dataset file
echo "frames $(ls $FRAME_DIR | wc -l) -1" > $LIST_DIR/test_split1.txt
echo "frames $(ls $FRAME_DIR | wc -l) -1" > $LIST_DIR/train_split1.txt

# 3. Run feature extraction
bash ext_feats.sh $TMPDIR $LIST_DIR $FEAT_FILE

# 4. Get the class
python get_class.py $FEAT_FILE ../data/hmdb51/train_test_lists/actions.txt
