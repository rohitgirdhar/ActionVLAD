cd ../
python \
  eval_image_classifier.py \
  --gpus 3 \
  --batch_size 1 \
  --frames_per_video 25 \
  --checkpoint_path models/PreTrained/ActionVLAD-pretrained/charadesV1 \
  --dataset_dir data/charadesV1/frames \
  --dataset_list_dir data/charadesV1/train_test_lists \
  --dataset_name charades \
  --dataset_split_name test \
  --model_name inception_v2_tsn \
  --modality rgb \
  --num_readers 4 \
  --num_preprocessing_threads 4 \
  --preprocessing_name vgg_ucf \
  --bgr_flip True \
  --pooling netvlad \
  --netvlad_initCenters 64 \
  --num_streams 1 \
  --split_id 1 \
  --store_feat stream0/logits \
  --store_feat_path  data/charadesV1/feats.h5
