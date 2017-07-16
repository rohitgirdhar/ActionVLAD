H5_FILE=$1
if [ $# -lt 1 ]; then
  echo "Usage: ./prog <h5 file>"
fi

# fixed paths
CHARADES_TEST_FILE="data/charadesV1/train_test_lists/test_split1.txt"
CHARADES_CODE_DIR="data/charadesV1/official_release"

# first convert the format and store into a temp file
TMP_FPATH=$(mktemp)
python eval/charades_convert_scores_format.py \
  --scores $H5_FILE \
  --test $CHARADES_TEST_FILE \
  --outfpath $TMP_FPATH

# now compute mAP
matlab -nodisplay -r "cd $CHARADES_CODE_DIR; Charades_v1_classify('$TMP_FPATH', '$CHARADES_CODE_DIR/Charades_v1_test.csv'); exit;"
rm $TMP_FPATH
