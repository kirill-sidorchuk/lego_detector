import cv2
import numpy as np
import argparse
# from Finetune import IMAGE_HEIGHT, IMAGE_WIDTH
# from ModelUtils import create_model_class


def video_capture(args):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('chaplin.mp4')

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)

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
    parser.add_argument("camera", type=int, default=0, help="camera device index")
    parser.add_argument("--tta", type=int, default=0, help="0 - no TTA, 1 - hflip, 2 - vflip+hflip")
    parser.add_argument("--rtta", type=int, default=0, help="Robot TTA. <1 - no TTA, >1 - number of images to take for TTA")
    parser.add_argument("--tta_mode", type=str, default="mean", help="'mean' or 'majority' voting TTA")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--snapshot", type=str, help="snapshot")

    _args = parser.parse_args()
    video_capture(_args)