# [ActionVLAD: Learning spatio-temporal aggregation for action classification](https://rohitgirdhar.github.io/ActionVLAD/)

If this code helps with your work/research, please consider citing

Rohit Girdhar, Deva Ramanan, Abhinav Gupta, Josef Sivic and Bryan Russell.
**ActionVLAD: Learning spatio-temporal aggregation for action classification**.
In Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

```txt
@inproceedings{Girdhar_17a_ActionVLAD,
    title = {{ActionVLAD}: Learning spatio-temporal aggregation for action classification},
    author = {Girdhar, Rohit and Ramanan, Deva and Gupta, Abhinav and Sivic, Josef and Russell, Bryan},
    booktitle = {CVPR},
    year = 2017
}
```

## Updates

- July 15, 2017: Released Charades models
- May 7, 2017: First release

## Quick Fusion
If you're only looking for our final last-layer features that can be combined with your method, we provide those
for the following datasets:

1. HMDB51: `data/hmdb51/final_logits`
2. Charades v1: `data/charadesV1/final_logits`

Note: Be careful to re-organize them given our filename and class ordering.

## Docker installation

Create docker_files folder where there should be the cudnn5.1 (include and lib) and also the models folder.
```
$ docker build -t action:latest .
```


## Pre-requisites
This code has been tested on a Linux (CentOS 6.5) system, though should be compatible with any OS running python and tensorflow.

1. TensorFlow (0.12.0rc0)
   - There have been breaking API changes in v1.0, so this code is not directly compatible with the latest tensorflow release. 
     You can try to use my pre-compiled [WHL file](https://cmu.box.com/shared/static/ayc9oeuwrmi5dnamdrz99n63bwnznely.whl).
   - You may consider installing tensorflow into an environment. On anaconda, it can be done by:

        ```bash
        $ conda create --name tf_v0.12.0rc0
        $ source activate tf_v0.12.0rc0
        $ conda install pip  # need to install pip into this env,
                             # else it will use global pip and overwrite your
                             # main TF installation
        $ pip install h5py  # and other libs, if need to be installed
        $ git clone https://github.com/tensorflow/tensorflow.git
        $ git checkout tags/0.12.0rc0
        $ # Compile tensorflow. Refer https://www.tensorflow.org/install/install_sources
        $ # If compiling on a CentOS (<7) machine, you might find the following instructions useful:
        $ # http://rohitgirdhar.github.io/tools/2016/12/23/compile-tf.html
        $ pip install --upgrade --ignore-installed /path/to/tensorflow_pkg.whl
        ```

2. Standard python libraries
   - pip
   - scikit-learn 0.17.1
   - h5py
   - pickle, cPikcle etc


## Quick Demo

This demo runs the RGB ActionVLAD model on a video. You will need the pretrained
models, which can be downloaded using the `get_models.sh` script, as described later
in this document.

```bash
$ cd demo
$ bash run.sh <video_path>
```

## Setting up the data


The videos need to be stored on disk as individual frame JPEG files, and similarly for optical flow.
The list of train/test videos are specified by text files, similar to the one in
`data/hmdb51/train_test_lists/train_split1.txt`. Each line consists of:

```txt
video_path number_of_frames class_id
```

Sample train/test files are in `data/hmdb51/train_test_lists`. The frames must be named in format: `image_%05d.jpg`.
Flow is stored similarly, with 2(n-1) files per video than the frames (n), named as `flow_%c_%05d.jpg`, where the
`%c` corresponds to `x` and `y`. This follows the data style followed in
various [previous works](http://yjxiong.me/others/action_recog/).

NOTE: For HMDB51, I renamed the videos to avoid issues with special characters in the filenames,
and hence the numbers in the train/test files.
The list of actual filenames is provided in `data/hmdb51/train_test_lists/AllVideos.txt`, and the new
name for each video in that list is the 1-indexed line number of that video.
The `AllVideos_renamed.txt` contains all the HMDB videos that are a part of one or all of the train/test splits
(it has fewer entries than `AllVideos.txt` because some videos are not in any split). So, the video `brush_hair/19`
in that file (and in the train/test split files) would correspond to the line number 19 in `AllVideos.txt`.

Create soft links to the directories where the frames are stored as following, so the provided scripts work out-of-the-box.

```bash
$ ln -s /path/to/hmdb51/frames data/hmdb51/frames
$ ln -s /path/to/hmdb51/flow data/hmdb51/flow
```

and so on. Since the code requires random access to this data
while training, it is advisable to store the frames/flow on a
fast disk/SSD.

For ease of reproduction, you can download our [frames](https://cmu.box.com/shared/static/i3q01shr30ziccf4b500g16t3podvsor.tgz) (`.tgz`, 9.3GB) and
[optical flow](https://cmu.box.com/shared/static/prpeizkk9ohil8yx40cdlodp84u8ttva.tgz) (`.tgz`, 4.7GB) on HMDB51.
Our UCF101 models should be compatible with the data provided with the [Good Practices](http://yjxiong.me/others/action_recog/) paper.


### Charades Data

Can be directly downloaded from [official website](http://allenai.org/plato/charades/).
This code assumes the [480px scaled frames](http://ai2-website.s3.amazonaws.com/data/Charades_v1_480.zip)
to be stored at `data/charadesV1/frames`.

## Testing pre-trained models

Download the models using `get_models.sh` script. Comment out specific lines
to download a subset of models.

Test all the models using the following scripts:

```bash
$ cd experiments
$ bash ext_all_logits.sh  # Stores all the features for each split
$ bash combine_streams.sh <split_id>  # change split_id to get final number for each split.
```

The above scripts (with provided models) should reproduce the following performance. The
iDT features are available from [Varol16]. You can also run these with the pre-computed
features provided in the `data/` folder.

| Split  | RGB | Flow | Combined (1:2) | iDT[Varol16] | ActionVLAD+iDT |
|--------|-----|------|----------|------|-----|
| 1      | 51.4 | 59.0 | 66.7 |  56.7 | 70.1 |
| 2      | 49.2 | 59.7 | 66.5 | 57.2 | 69.0 |
| 3      | 48.6 | 60.6 | 66.3 | 57.8 | 70.1 |
| Avg    | 49.7 | 59.8 | 66.5 | 57.2 | 69.7 |

NOTE: There is very small difference (<0.1%) in the final numbers above from what's reported in the paper.
This was due to an [undocumented behavior of tensorflow `tf.train.batch` functionality](https://github.com/tensorflow/tensorflow/issues/9441),
which is slightly non-deterministic when used with multiple threads.
This can lead to some local shuffling in the order of videos at test time, which
leads to inconsistent results when late-fusing different methods.
This has been fixed now by
forcing the use of a single thread when saving features to the disk.

### Charades testing
Charades models were trained using a slightly different version of TF, so need a
bit more work to test. Download the model data file as mentioned
in the `get_data.sh` script (by default, it will download).
Then,

```bash
$ cp models/PreTrained/ActionVLAD-pretrained/charadesV1/checkpoint.example models/PreTrained/ActionVLAD-pretrained/charadesV1/checkpoint
$ vim models/PreTrained/ActionVLAD-pretrained/charadesV1/checkpoint
$ # modify the file and replace the $BASE_DIR with the **absolute path** of where the ActionVLAD repository is cloned to
$ # Now, for testing
$ cd experiments && bash 006_InceptionV2TSN_RGB_Charades_eval.sh
$ cd .. && bash eval/charades_eval.sh data/charadesV1/feats.h5
```

The above should reproduce the following numbers:

|      | mAP | wAP |
|------|-----|-----|
| ActionVLAD (RGB) | 17.66 | 25.17 |

## Training

Note that in the following training steps, RGB model is trained directly on top of ImageNet
initialization while the flow models are trained over the flow stream of a two-stream
model. This is just because we found that training the last few layers in RGB
stream (of a two-stream model) gets  good enough performance, so everything before and including conv5_3
is left untouched to the imagenet initialization. Since we build our model
on top of conv5_3, we end up essentially training on top of ImageNet initialization.


### RGB model

```bash
$ ### Initialization for ActionVLAD (KMeans)
$ cd experiments
$ bash 001_VGG_RGB_HMDB_netvlad_feats_for_clustering.sh  # extract random subset of features
$ 001_VGG_RGB_HMDB_netvlad_cluster.sh  # cluster the features to initialize ActionVLAD
$ ### Training the model
$ bash 001_VGG_RGB_HMDB_netvlad_stage1.sh  # trains the last layer with fixed ActionVLAD
$ bash 001_VGG_RGB_HMDB_netvlad_stage2.sh  # trains the last layer+actionVLAD+conv5
$ bash 001_VGG_RGB_HMDB_netvlad_eval.sh  # evaluates the final trained model
```

### Flow model

```bash
$ ### Initialization for ActionVLAD (KMeans)
$ cd experiments
$ bash 001_VGG_Flow_HMDB_netvlad_feats_for_clustering.sh  # extract random subset of features
$ 001_VGG_Flow_HMDB_netvlad_cluster.sh  # cluster the features to initialize ActionVLAD
$ ### Training the model
$ bash 001_VGG_Flow_HMDB_netvlad_stage1.sh  # trains the last layer with fixed ActionVLAD
$ bash 001_VGG_Flow_HMDB_netvlad_stage2.sh  # trains the last layer+actionVLAD+conv5
$ bash 001_VGG_Flow_HMDB_netvlad_eval.sh  # evaluates the final trained model
```


## Miscellaneous

### Two-stream models

The following scripts run testing on the flow stream of our two-stream models.
As mentioned earlier, we didn't need a RGB stream model for ActionVLAD training
since we could train directly on top of ImageNet initialization.

```bash
$ cd experiments
$ bash 005_VGG_Flow_HMDB_TestTwoStream.sh
```

You can also train two-stream models using this code base. Here's a sample script
to train a RGB stream (not tested, so might require playing around with hyperparameters):

```bash
$ cd experiments
$ bash 005_VGG_RGB_HMDB_TrainTwoStream.sh
$ bash 005_VGG_RGB_HMDB_TestTwoStream.sh
```

## References

[Varol16]: Gul Varol, Ivan Laptev and Cordelia Schmid.
[Long-term Convolutions for Action Recognition.](https://www.di.ens.fr/willow/research/ltc/)
arXiv 2016.


## Acknowledgements
This code is based on the [tensorflow/models](https://github.com/tensorflow/models/tree/master/slim) repository,
so thanks to the original authors/maintainers for releasing the code.
