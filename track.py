import os
import glob

import numpy as np
import cv2

from mv_extractor import VideoCap
from mvt.visu import draw_motion_vectors, draw_boxes

from detector import DetectorTF
from mvt.loaders import load_detections

from mvt.tracker import MotionVectorTracker


if __name__ == "__main__":

    # example video file (taken from MOT17 benchmark)
    codec = "mpeg4"  # mpeg4 or h264
    scaling_factor = 1.0  # always 1.0
    video_file = "data/MOT17-09-FRCNN-{}-{}.mp4".format(codec, scaling_factor)

    # offline detections for the example video
    use_offline_detections = False
    detections_file = "data/det.txt"
    num_frames = 525  # number of frames in the example video (only needed for offline detections)

    # specify object detector and tracker settings
    detector_path = "models/detector/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb"  # detector frozen inferenze graph (*.pb)
    detector_box_size_thres = None #(0.25*1920, 0.6*1080) # discard detection boxes larger than this threshold
    detector_interval = 20
    tracker_iou_thres = 0.3
    det_conf_threshold = 0.8  # set to 0.5 for DPMv5 and to 0.9 for FRCNN
    state_thresholds = (0, 1, 10)

    # When True you have to press 's' key to advance the video by one frame. If False the video just plays.
    step_wise = False

    show_detector_output = False
    show_previous_boxes = True
    show_motion_vectors = True

    tracker_baseline = MotionVectorTracker(
        iou_threshold=tracker_iou_thres,
        det_conf_threshold=det_conf_threshold,
        state_thresholds=state_thresholds,
        use_only_p_vectors=False,
        use_numeric_ids=True,
        measure_timing=True)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 640, 360)

    if use_offline_detections:
        det_boxes_all, det_scores_all = load_detections(detections_file, num_frames)
    else:
        detector = DetectorTF(path=detector_path,
                            box_size_threshold=detector_box_size_thres,
                            scaling_factor=1.0,
                            gpu=0)

    cap = VideoCap()

    ret = cap.open(video_file)
    if not ret:
        raise RuntimeError("Could not open the video file")

    frame_idx = 0

    # box colors
    color_detection = (0, 0, 150)
    color_tracker_baseline = (0, 0, 255)
    color_previous_baseline = (150, 150, 255)

    prev_boxes_baseline = None

    while True:
        ret, frame, motion_vectors, frame_type, _ = cap.read()
        if not ret:
            print("Could not read the next frame")
            break

        # draw entire field of motion vectors
        if show_motion_vectors:
            frame = draw_motion_vectors(frame, motion_vectors)

        # draw info
        frame = cv2.putText(frame, "Frame Type: {}".format(frame_type), (1000, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # draw color legend
        frame = cv2.putText(frame, "Detection", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_detection, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Baseline Previous Prediction", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_previous_baseline, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Baseline Tracker Prediction", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_tracker_baseline, 2, cv2.LINE_AA)

        # update with detections
        if frame_idx % detector_interval == 0:
            if use_offline_detections:
                det_boxes = det_boxes_all[frame_idx] * scaling_factor
                det_scores = det_scores_all[frame_idx] * scaling_factor
            else:
                detections = detector.detect(frame)
                det_boxes = detections['detection_boxes']
                det_scores = detections['detection_scores']
            tracker_baseline.update(motion_vectors, frame_type, det_boxes, det_scores)
            if prev_boxes_baseline is not None and show_previous_boxes:
               frame = draw_boxes(frame, prev_boxes_baseline, color=color_previous_baseline)
            prev_boxes_baseline = np.copy(det_boxes)

        # prediction by tracker
        else:
            tracker_baseline.predict(motion_vectors, frame_type)
            track_boxes_baseline = tracker_baseline.get_boxes()
            box_ids_baseline = tracker_baseline.get_box_ids()

            if prev_boxes_baseline is not None and show_previous_boxes:
               frame = draw_boxes(frame, prev_boxes_baseline, color=color_previous_baseline)
            prev_boxes_baseline = np.copy(track_boxes_baseline)

            frame = draw_boxes(frame, track_boxes_baseline, box_ids=box_ids_baseline, color=color_tracker_baseline)

        if show_detector_output:
            frame = draw_boxes(frame, det_boxes, scores=det_scores, color=color_detection)

        # print FPS
        print("Frame {}, FPS Predict {}, FPS Update {}".format(frame_idx,
            1/tracker_baseline.last_predict_dt, 1/tracker_baseline.last_update_dt))

        frame_idx += 1
        cv2.imshow("frame", frame)

        # handle key presses
        # 'q' - Quit the running program
        # 's' - enter stepwise mode
        # 'a' - exit stepwise mode
        key = cv2.waitKey(1)
        if not step_wise and key == ord('s'):
            step_wise = True
        if key == ord('q'):
            break
        if step_wise:
            while True:
                key = cv2.waitKey(1)
                if key == ord('s'):
                    break
                elif key == ord('a'):
                    step_wise = False
                    break

    cap.release()
    cv2.destroyAllWindows()
