import cv2
import numpy as np
import argparse


# from Finetune import IMAGE_HEIGHT, IMAGE_WIDTH
# from ModelUtils import create_model_class
from Predict import predict_with_tta, load_model


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

    # loading model
    print("loading model...")
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
    parser = argparse.ArgumentParser(description='Finetune model')
    parser.add_argument("data_root", type=str, help="data root dir")
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    parser.add_argument("--tta", type=int, default=0, help="0 - no TTA, 1 - hflip, 2 - vflip+hflip")
    parser.add_argument("--rtta", type=int, default=0,
                        help="Robot TTA. <1 - no TTA, >1 - number of images to take for TTA")
    parser.add_argument("--tta_mode", type=str, default="mean", help="'mean' or 'majority' voting TTA")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--snapshot", type=str, help="snapshot")

    _args = parser.parse_args()
    video_capture(_args)
