# Lego parts classifier

## Workflow
* Collect data: make photos of multiple parts in each image
* Prepare dataset: segment out individual parts and sort the results
* Finalize dataset: create train/validation split
* Train a model: finetune existing model trained on ImageNet
* Test model

## Python
We are using Python 3 to be able to run Tensorflow on Windows

## Dataset directory structure
dataset root/
* raw - raw photos from camera (user input)
* downsampled - raw photos downsampled to 1024 px on longer side (generated from raw photos)
* masks - masks generated from downsamled raw images (these are 4-value masks for GrabCut algorithm). You can edit masks in graphical editor to improve segmentation.
* segmentation - segmentation results of GrabCut algorithm - all background pixels are zeroed.
* parts - contains extracted individual parts for each image
* sorted - user sorted parts. Each subdirectory contains parts belonging to the same class. Name of subdirectory is regarded as class label name
* test - raw photos to use as a test set. File names are regarded as true labels.

## Workflow commandlines

<b>python3 PrepareDataset.py <i>data_root_dir</i></b>\
Runs segmentation step. The procedure is as follows:\
For each raw image in 'raw' directory
* create downsampled image if not exists already
* create foreground segmentation 4-value mask for GrabCut if not exists already in 'masks' dir
* run GrabCut to segment out background (if segmentation result does not exist already) 
* extract individual parts from segmentation results (always overwrites results)

<b>python3 FinalizeDataset.py <i>data_root_dir</i></b>\
Creates train/validation split using 'sorted' directory.
Results - train.txt and val.txt - are created in data root dir.

<b>python3 Finetune.py --model <i>A</i> --snapshot <i>weights-784-0.969.hdf5</i></b>\
Runs training.\
--model specifies model name to use.\
--snapshot <i>file</i> [optional] specifies a snapshot file to restart training from.\
--debug_epochs <i>N</i> [optional] specifies number of training epochs to save images fed to network. These images will be written in data_root_dir/debug directory.
 
<b>python3 Predict.py new_test\sorted measure --tta 2 --rtta 0 --model A --snapshot weights-784-0.969.hdf5</b>\
Runs prediction on test set (from 'new_test\sorted' subdirectory).\
There are two modes: 'measure' and 'sort'
* measure will calculate top1 and top5 accuracies given images with known labels, stored to directories (one directory - one class). Classification results and final accuracies are written to console.

* sort will sort images with unknown labels into directories - dir name will correspond to predicted class label.

--model and --snapshot arguments are the same as for 'Finetune.py' script.\
--tta 0 means no test time data augmentation, 1 - do vertical flip, 2 - do vertical and horizontal flips.\
--rtta <1 means no robot test time data augmentation, 2 and more means take that many images from sorted directory and average results. 
--tta_mode 'mean' or 'majority' aggregation method for TTA. Default is 'mean'.


## Results

These results were obtained using command line params as follows:\
--tta 3 --rtta 3 --tta_mode mean\
Top 1 and top 5 accuracies were measured on a test set of 386 files

#### ResNet50 (24M params)
Best snapshot: model C, weights-860-0.917.hdf5\
top1: 91.4%\
top5: 95.7\
predict time per image: 900ms 

#### MobileNet (5M params)
Best snapshot: model E, weights-745-0.875.hdf5\
top1: 80.7%\
top5: 94.6%\
predict time per image: 347 ms

#### InceptionV3 (M params)
Best snapshot: model F, .hdf5\
top1: \
top5: \
predict time per image: ms\
