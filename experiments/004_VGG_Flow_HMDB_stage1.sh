cd ../
LD_PRELOAD=/usr/lib64/libtcmalloc.so.4 \
  $(which python) \
  train_image_classifier.py \
  --batch_size 5 \
  --gpus 3 \
  --frames_per_video 25 \
  --iter_size 2 \
  --checkpoint_path models/PreTrained/2-stream-pretrained/flow_split1.ckpt \
  --train_dir models/Experiments/004_VGG_Flow_HMDB_stage1 \
  --dataset_list_dir data/hmdb51/train_test_lists/ \
  --dataset_dir data/hmdb51/flow \
  --dataset_name hmdb51 \
  --dataset_split_name train \
  --model_name vgg_16 \
  --modality flow10 \
  --num_readers 2 \
  --num_preprocessing_threads 4 \
  --learning_rate 0.01 \
  --optimizer adam \
  --opt_epsilon 1e-4 \
  --preprocessing_name vgg_ucf \
  --bgr_flip False \
  --pooling netvlad \
  --netvlad_initCenters models/kmeans-init/hmdb51/flow_conv5_kmeans64_split1.pkl \
  --classifier_type linear \
  --trainable_scopes stream0/classifier \
  --pooled_dropout 0.5 \
  --num_steps_per_decay 5000 \
  --learning_rate_decay_factor 0.1 \
  --clip_gradients 5 \
  --log_every_n_steps 10 \
  --num_streams 1 \
  --max_number_of_steps 8000 \
  --checkpoint_exclude_scopes stream0/NetVLAD,stream0/classifier \
