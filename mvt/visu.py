import cv2
import numpy as np


def draw_motion_vectors(frame, motion_vectors, format='torch'):
    """Draw motion vectors onto the frame."""
    if np.shape(motion_vectors)[0] > 0:
        num_mvs = np.shape(motion_vectors)[0]
        for mv in np.split(motion_vectors, num_mvs):
            start_pt = (mv[0, 5], mv[0, 6])
            end_pt = (mv[0, 3], mv[0, 4])
            if mv[0, 0] < 0:
                frame = cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.3)
            else:
                frame = cv2.arrowedLine(frame, start_pt, end_pt, (0, 255, 0), 1, cv2.LINE_AA, 0, 0.3)
    return frame


def draw_boxes(frame, bounding_boxes, box_ids=None, scores=None, color=(0, 255, 0)):
    for i, box in enumerate(bounding_boxes):
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[0] + box[2])
        ymax = int(box[1] + box[3])
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_4)
        if box_ids is not None:
            frame = cv2.putText(frame, '{}'.format(str(box_ids[i])[:6]), (xmin, ymin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        if scores is not None:
            frame = cv2.putText(frame, '{}'.format(str(scores[i])), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return frame
