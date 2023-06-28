import cv2 as cv
import numpy as np
from onemetric.cv.utils.iou import box_iou_batch
from ultralytics.yolo.utils.plotting import Annotator


def read_frame(camera):
    ok, frame = camera.read()

    if not ok:
        print("ERROR: Can't receive frame!")
        exit()
    frame_resized = cv.resize(frame, (640, 480))

    return frame_resized


# Возвращает в формате х1 у1 х2 у2 p label
def get_detections(predictions_model, frame):
    results = predictions_model.predict(frame)

    detections = []
    labels = []

    for r in results:
        boxes = r.boxes
        obj_count = boxes.shape[0]
        for box in boxes:
            detection = []
            b = box.xyxy[0]
            c = box.cls
            p = box.conf
            x1, y1, x2, y2 = b.tolist()
            label = predictions_model.names[int(c)] + ' ' + str(np.round(float(p), 2) * 100) + '%'
            detection = [x1, y1, x2, y2, float(p[0])]
            if p > 0.6:
                detections.append(detection)
                labels.append(label)

    #     if not detections:
    #         detections.append([])

    return np.array(detections), np.array(labels)


# Возвращает в формате х1 у1 х2 у2
def get_tracks(tracker, detections, frame):
    detections_for_track = detections[:, :5]
    tracker_bboxes = tracker.update(output_results=detections_for_track, \
                                    img_info=frame.shape, \
                                    img_size=frame.shape)
    bbox_coords = []

    for tracker_bbox in tracker_bboxes:
        x1, y1, w, h = tracker_bbox.tlwh
        bbox_coord = [x1, y1, x1 + w, y1 + h]
        bbox_coords.append(bbox_coord)

    return np.array(bbox_coords)


def match_detections_with_tracks(detections, tracks):
    detection_boxes = detections[:, :4]
    tracks_boxes = tracks

    iou = box_iou_batch(tracks_boxes, detection_boxes)
    track2detection = np.argmax(iou, axis=1)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            detection_boxes[detection_index] = tracks[tracker_index]
    return detection_boxes


def annotate_frames(frame, boxes_fin, labels):
    annotator = Annotator(frame)
    for i, box in enumerate(boxes_fin):
        if i + 1 <= len(labels):
            annotator.box_label(box, labels[i])