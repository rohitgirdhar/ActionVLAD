SPLIT_ID=$1
cd ..
IDT_PATH=models/PreTrained/iDT/hmdb51/split${SPLIT_ID}.h5
if [ -f $IDT_PATH ]; then
  idt_cmd="--idt_scores $IDT_PATH --idt_wt 0.25"
fi
python combine_streams.py \
  -s /tmp/rgb_split${SPLIT_ID}.h5 \
  -t /tmp/flow_split${SPLIT_ID}.h5 \
  -f data/hmdb51/train_test_lists/test_split${SPLIT_ID}.txt \
  $idt_cmd
