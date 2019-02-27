import os

import torch

import cv2
import numpy as np
import argparse


# from Finetune import IMAGE_HEIGHT, IMAGE_WIDTH
# from ModelUtils import create_model_class
from Predict import predict_with_tta, load_model

from ImageDataset import ImageDataset
from modelA import modelA


def clip(frame):
    width = frame.shape[1]
    height = frame.shape[0]
    x0 = width//4
    y0 = height//4
    new_width = width//2
    new_height = height//2
    center = frame[y0:y0+new_height, x0:x0+new_width].copy()
    return center


def video_capture(args):

    use_cuda = torch.cuda.is_available() and args.gpu.lower() == 'true'
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using GPU" if use_cuda else "Using CPU")

    # loading train foregrounds
    train_foregrounds = ImageDataset(os.path.join(args.data_root, "train.txt"), "sorted")

    # loading model
    print("loading model...")
    model = modelA(num_classes, False).to(device)
    model_name = type(model).__name__

    model, labels_map, int_to_labels_map = load_model(args.data_root, args.model, args.snapshot)

    print("start video capture...")
    cap = cv2.VideoCapture(args.camera)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX

    past_frames = []

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            past_frames.append(clip(frame))
            if len(past_frames) > args.rtta:
                past_frames.pop(0)

            frame_to_show = cv2.resize(past_frames[-1], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

            probs = predict_with_tta(args.tta, True, past_frames, model)[0]
            sorted_indexes = np.argsort(probs)
            for t in range(5):
                label_index = sorted_indexes[-t - 1]
                label_prob = probs[label_index]
                label_name = int_to_labels_map[label_index]
                s = "%1.2f%% %s" % (label_prob * 100.0, label_name)
                cv2.putText(frame_to_show, s, (0, (t+1)*32), font, 1, (0,0,180), 2, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('Frame', frame_to_show)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prediction from web camera')
    parser.add_argument("model", type=str, default=None, help="Name of the model to train.")
    parser.add_argument("--data_root", type=str, default=None, help="Path to data.")
    parser.add_argument("--snapshot", type=int, default=None, help="Iteration to start from snapshot.")
    parser.add_argument("--gpu", type=str, default='true', help="Set to 'true' for GPU.")

    _args = parser.parse_args()
    video_capture(_args)
