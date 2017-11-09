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
 
<b>python3 Predict.py --model <i>A</i> --snapshot <i>weights-784-0.969.hdf5</i></b>\
Runs prediction on test set (from 'test' subdirectory).\
--model and --snapshot arguments are the same as for 'Finetune.py' script.\
Classification results and final accuracies are written to console.
 