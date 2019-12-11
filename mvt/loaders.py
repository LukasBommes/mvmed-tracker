import numpy as np


def load_detections(det_file, num_frames):
    det_boxes = []
    det_scores = []
    raw_data = np.genfromtxt(det_file, delimiter=',')
    for frame_idx in range(num_frames):
        idx = np.where(raw_data[:, 0] == frame_idx+1)
        if idx[0].size:
            det_boxes.append(np.stack(raw_data[idx], axis=0)[:, 2:6])
            det_scores.append(np.stack(raw_data[idx], axis=0)[:, 6])
        else:
            det_boxes.append(np.empty(shape=(0, 4)))
            det_scores.append(np.empty(shape=(0,)))
    return det_boxes, det_scores
