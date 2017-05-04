# note, can use the any split model as upto conv5 for RGB (on HMDB) it is basically imagenet pretrained
cd ../
LD_PRELOAD=/usr/lib64/libtcmalloc.so.4 \
  $(which python) \
  eval_image_classifier.py \
  --gpus 2 \
  --batch_size 64 \
  --frames_per_video 1 \
  --max_num_batches 100 \
  --checkpoint_path models/PreTrained/2-stream-pretrained/hmdb51/flow/split1.ckpt \
  --dataset_dir data/hmdb51/flow \
  --dataset_list_dir data/hmdb51/train_test_lists/ \
  --dataset_name hmdb51 \
  --dataset_split_name train \
  --model_name vgg_16 \
  --modality flow10 \
  --num_readers 4 \
  --num_preprocessing_threads 4 \
  --preprocessing_name vgg_ucf \
  --bgr_flip False \
  --pooling None \
  --store_feat stream0/vgg_16/conv5/conv5_3 \
  --store_feat_path models/Experiments/001_VGG_RGB_HMDB_netvlad_stage1/Features/flow_conv5.h5 \
  --force_random_shuffle True \
  --num_streams 1 \
  --split_id 1
