cd ..
LD_PRELOAD=/usr/lib64/libtcmalloc.so.4 \
  python \
  train_image_classifier.py \
  --batch_size 4 \
  --gpus 0,1,2,3 \
  --frames_per_video 1 \
  --iter_size 1 \
  --checkpoint_path models/PreTrained/imagenet-trained-CUHK/vgg_16_action_rgb_pretrain.caffemodel.npy \
  --var_name_mapping cuhk-action-vgg \
  --train_dir models/Experiments/005_VGG_RGB_HMDB_TrainTwoStream \
  --dataset_list_dir data/hmdb51/train_test_lists/ \
  --dataset_dir data/hmdb51/frames \
  --dataset_name hmdb51 \
  --dataset_split_name train \
  --model_name vgg_16 \
  --modality rgb \
  --num_readers 4 \
  --num_preprocessing_threads 4 \
  --learning_rate 0.001 \
  --optimizer momentum \
  --preprocessing_name vgg_ucf \
  --bgr_flip True \
  --num_steps_per_decay 2000 \
  --learning_rate_decay_factor 0.1 \
  --num_streams 1 \
  --trainable_scopes stream0/classifier,stream0/vgg_16/fc7,stream0/vgg_16/fc6 \
  --train_image_size 224 \
  --weight_decay 5e-4 \
  --split_id 1 \
  --max_number_of_steps 5000
