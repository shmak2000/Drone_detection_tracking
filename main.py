import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
display.clear_output()
import ultralytics
from ultralytics import YOLO
from IPython.display import display, Image
import torch
from ultralytics.yolo.utils.plotting import Annotator
from yolox.tracker.byte_tracker import BYTETracker
from onemetric.cv.utils.iou import box_iou_batch
from ByteTrackerArgs import BYTETrackerArgs

from functions import read_frame, get_detections, get_tracks, \
    match_detections_with_tracks, annotate_frames


ultralytics.checks()
print('CUDA is available' if torch.cuda.is_available() \
      else 'ERROR: CUDA is NOT available')

# Insert video file + yolo weights (check readme file for my weights)
cap = cv.VideoCapture("videoplayback.mp4")
model = YOLO("best.pt")
tracker = BYTETracker(BYTETrackerArgs)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    frame = read_frame(cap)

    detections, labels = get_detections(model, frame)

    if detections.shape[0]:
        tracks = get_tracks(tracker, detections, frame)

        if tracks.shape[0]:
            boxes_fin = match_detections_with_tracks(detections, tracks)

        annotate_frames(frame, boxes_fin, labels)

    cv.imshow('frame', frame)

    if cv.waitKey(1) == 27:
        print("Process finished by User!")
        break

cap.release()
cv.destroyAllWindows()