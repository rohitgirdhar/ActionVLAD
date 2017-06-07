cd ../
LD_PRELOAD=/usr/lib64/libtcmalloc.so.4 \
  python \
  eval_image_classifier.py \
  --gpus 0 \
  --batch_size 1 \
  --frames_per_video 25 \
  --checkpoint_path models/PreTrained/ActionVLAD-pretrained/hmdb51/rgb/split1.ckpt \
  --dataset_dir $1 \
  --dataset_list_dir $2 \
  --dataset_name hmdb51 \
  --dataset_split_name test \
  --model_name vgg_16 \
  --modality rgb \
  --num_readers 4 \
  --num_preprocessing_threads 4 \
  --preprocessing_name vgg_ucf \
  --bgr_flip True \
  --pooling netvlad \
  --netvlad_initCenters 64 \
  --classifier_type linear \
  --ncrops 5 \
  --store_feat stream0/logits \
  --store_feat_path $3 \
