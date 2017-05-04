FEAT_DIR="/tmp/"
for split_id in {1..3}; do
  for mod in rgb flow; do
    bash ext_logits.sh $mod $split_id $FEAT_DIR/${mod}_split${split_id}.h5 hmdb51
  done
done
