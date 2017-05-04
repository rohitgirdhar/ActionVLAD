cd ..
LD_PRELOAD=/usr/lib64/libtcmalloc.so.4 \
  python \
  train_image_classifier.py \
  --batch_size 4 \
  --gpus 2,3 \
  --frames_per_video 25 \
  --iter_size 2 \
  --checkpoint_path models/PreTrained/imagenet-trained-CUHK/vgg_16_action_rgb_pretrain_uptoConv5.ckpt \
  --checkpoint_style v2_withStream \
  --train_dir models/Experiments/001_VGG_RGB_HMDB_netvlad_stage1 \
  --dataset_list_dir data/hmdb51/train_test_lists/ \
  --dataset_dir data/hmdb51/frames \
  --dataset_name hmdb51 \
  --dataset_split_name train \
  --model_name vgg_16 \
  --modality rgb \
  --num_readers 4 \
  --num_preprocessing_threads 4 \
  --learning_rate 0.01 \
  --optimizer adam \
  --opt_epsilon 1e-4 \
  --preprocessing_name vgg_ucf \
  --bgr_flip True \
  --pooling netvlad \
  --netvlad_initCenters models/Experiments/001_VGG_RGB_HMDB_netvlad_stage1/Features/imnet_conv5_kmeans64.pkl \
  --pooled_dropout 0.5 \
  --num_steps_per_decay 5000 \
  --learning_rate_decay_factor 0.1 \
  --clip_gradients 5 \
  --num_streams 1 \
  --trainable_scopes stream0/classifier \
  --checkpoint_exclude_scopes stream0/NetVLAD,stream0/classifier \
  --train_image_size 224 \
  --weight_decay 4e-5 \
  --split_id 1 \
  --max_number_of_steps 10000
