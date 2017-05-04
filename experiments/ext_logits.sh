MODALITY=$1
SPLIT_ID=$2
STORE_FEAT_PATH=$3
DATASET=$4
if [ "$MODALITY" == "rgb" ]; then
  BGR_FLIP="True"
  DIR="frames"
  MODALITY_NAME="rgb"
  NCROPS=1
else
  BGR_FLIP="False"
  DIR="flow"
  MODALITY_NAME="flow10"
  NCROPS=5
fi

cd ../
LD_PRELOAD=/usr/lib64/libtcmalloc.so.4 \
  python \
  eval_image_classifier.py \
  --gpus 1 \
  --batch_size 1 \
  --frames_per_video 25 \
  --checkpoint_path models/PreTrained/ActionVLAD-pretrained/${DATASET}/${MODALITY}/split${SPLIT_ID}.ckpt \
  --dataset_dir data/${DATASET}/$DIR \
  --dataset_list_dir data/${DATASET}/train_test_lists \
  --dataset_name hmdb51 \
  --dataset_split_name test \
  --model_name vgg_16 \
  --modality $MODALITY_NAME \
  --num_readers 4 \
  --num_preprocessing_threads 4 \
  --preprocessing_name vgg_ucf \
  --bgr_flip $BGR_FLIP \
  --pooling netvlad \
  --netvlad_initCenters 64 \
  --classifier_type linear \
  --ncrops $NCROPS \
  --split_id $SPLIT_ID \
  --store_feat stream0/logits \
  --store_feat_path $STORE_FEAT_PATH
