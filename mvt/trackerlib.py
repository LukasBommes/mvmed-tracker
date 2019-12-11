import numpy as np
#import cv2
import scipy
from scipy import optimize
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, Q_continuous_white_noise


def get_vectors_by_source(motion_vectors, source):
    """Returns subset of motion vectors with a specified source frame.

    The source parameter of a motion vector specifies the temporal position of
    the reference (source) frame relative to the current frame. Each vector
    starts at the point (src_x, sry_y) in the source frame and points to the
    point (dst_x, dst_y) in the current frame. If the source value is for
    example -1, then the reference frame is the previous frame.

    For B frames there are motion vectors which refer macroblocks both to past
    frames and future frames. By setting the source parameter to "past" this
    method filters out motion vectors referring to future frames and returns the
    set of motion vectors which refer to past frames (e.g. the equivalent to the
    motion vectors in P frames). Similarly, by setting the value to "future"
    only vectors referring to future frames are returned.

    Args:
        motion_vectors (`numpy.ndarray`): Array of shape (N, 11) containing all
            N motion vectors inside a frame. N = 0 is allowed meaning no vectors
            are present in the frame.

        source (`ìnt` or `string`): Motion vectors with this value for their
            source parameter (the location of the reference frame) are selected.
            If "future", all motion vectors with a positive source value are
            returned (only for B-frames). If "past" all motion vectors with
            a negative source value are returned.

    Returns:
        motion_vectors (`numpy.ndarray`): Array of shape (M, 11) containing all
        M motion vectors with the specified source value. If N = 0 => M = 0
        that is an empty numpy array of shape (0, 11) is returned.
    """
    if np.shape(motion_vectors)[0] == 0:
        return motion_vectors
    else:
        if source == "past":
            idx = np.where(motion_vectors[:, 0] < 0)[0]
        elif source == "future":
            idx = np.where(motion_vectors[:, 0] > 0)[0]
        else:
            idx = np.where(motion_vectors[:, 0] == source)[0]
        return motion_vectors[idx, :]


def normalize_vectors(motion_vectors):
    """Normalizes motion vectors to the past frame as reference frame.

    The source value in the first column is set to -1 for all p-vectors and
    set to 1 for all b-vectors. The x and y motion values are scaled
    accordingly. Vector source position and destination position are unchanged.

    Args:
        motion_vectors (`numpy.ndarray`): Array of shape (N, 11) containing all
            N motion vectors inside a frame. N = 0 is allowed meaning no vectors
            are present in the frame.

    Returns:
        motion_vectors (`numpy.ndarray`): Array of shape (M, 11) containing the
        normalized motion vectors. If N = 0 => M = 0 that is an empty numpy
        array of shape (0, 11) is returned.
    """
    if np.shape(motion_vectors)[0] == 0:
        return motion_vectors
    else:
        motion_vectors[:, 7] = motion_vectors[:, 7] / motion_vectors[:, 0]  # motion_x
        motion_vectors[:, 8] = motion_vectors[:, 8] / motion_vectors[:, 0]  # motion_y
        motion_vectors[:, 0] = np.sign(motion_vectors[:, 0])
        return motion_vectors


def get_nonzero_vectors(motion_vectors):
    """Returns subset of motion vectors which have non-zero magnitude.
    """
    if np.shape(motion_vectors)[0] == 0:
        return motion_vectors
    else:
        idx = np.where(np.logical_or(motion_vectors[:, 7] != 0, motion_vectors[:, 8] != 0))[0]
        return motion_vectors[idx, :]


def get_vectors_in_boxes(motion_vectors, boxes):
    """Returns a subset of motion vectors starting within a specific box.

    Given the full set of motion vectors inside a single frame and a query
    bounding box this function returns the subset of motion vectors which start
    inside the query box.

    Args:
        motion_vectors (`numpy.ndarray`): Array of shape (N, 11) containing all
            N motion vectors inside a frame. N = 0 is allowed meaning no vectors
            are present in the frame.

        boxes (`numpy.ndarray`): The K query bounding boxes as array of shape
            (K, 4) where each row represents the coordinates of a single
            bounding box [xmin, ymin, width, height].

    Returns:
        motion_vectors_subsets (`list` of `numpy.ndarray`): Each list entry
        corresponds to a bounding box in the boxes array and is an array of
        shape (M, 11) of all M motion vectors starting inside the
        corresponding box. If N = 0 or K = 0 an empty list is returned. If
        no motion vector falls into a bounding box the corresponding list
        entry if an empty numpy array of shape (0, 11).
    """
    if np.shape(motion_vectors)[0] == 0 or np.shape(boxes)[0] == 0:
        return []
    else:
        motion_vector_subsets = []
        for box in np.split(boxes, np.shape(boxes)[0]):
            box = box.reshape(4,)
            xmin = box[0]
            xmax = box[0] + box[2]
            ymin = box[1]
            ymax = box[1] + box [3]
            # get (x_src, y_src) considering possible scaling during normalization
            src_x = motion_vectors[:, 5] + (motion_vectors[:, 0] * motion_vectors[:, 7] / motion_vectors[:, 9])
            src_y = motion_vectors[:, 6] + (motion_vectors[:, 0] * motion_vectors[:, 8] / motion_vectors[:, 9])

            # get indices of vectors inside the box
            idx = np.where(np.logical_and(
                np.logical_and(src_x >= xmin, src_x <= xmax),
                np.logical_and(src_y >= ymin, src_y <= ymax)))[0]
            motion_vector_subsets.append(motion_vectors[idx, :])
        return motion_vector_subsets


def get_box_shifts(motion_vector_subsets, metric="median"):
    """Returns the shift of each box edge based on the contained motion vectors.

    This method computes the shifts of the bounding box from the motion vectors
    reference frame to the current frame. The shift is computed as mean of the
    x/y component of all motion vectors inside the bounding box. This method
    assumes normalized motion vectors.

    Args:
        motion_vector_subsets (`numpy.ndarray`): The motion vectors inside every
            box as returned by the `get_vectors_in_boxes` method.

        metric (`string`): Determines how the average components of the motion
            vectors inside each bounding box are computed. Can be either "mean"
            or "median".

    Returns:
        shifts (`numpy.ndarray`): Array of shape (K, 2) containing the desired
        shifts in pixels in x and y direction of each of the K query boxes.
        A negative sign indicates movement in the negative x or y direction
        (left and up). If N = 0 an empty numpy array of shape (0, 2) is
        returned.
    """
    num_boxes = np.shape(motion_vector_subsets)[0]
    shifts = np.zeros((num_boxes, 2))

    for i, mv_subset in enumerate(motion_vector_subsets):
        if np.shape(mv_subset)[0] > 0:

            # compute the vector components
            mvs_xc = mv_subset[:, 7] / mv_subset[:, 9]  # motion_x / (motion_scale * source)
            mvs_yc = mv_subset[:, 8] / mv_subset[:, 9]  # motion_y / (motion_scale * source)

            # compute edge shifts as weighted averages of the x/y components of all vectors
            if metric == "mean":
                shifts[i, 0] = np.mean(mvs_xc)  # x shift
                shifts[i, 1] = np.mean(mvs_yc)  # y shift
            elif metric == "median":
                shifts[i, 0] = np.median(mvs_xc)  # x shift
                shifts[i, 1] = np.median(mvs_yc)  # y shift
            else:
                raise ValueError("Invalid argument for metric provided. Use \"mean\" or \"median\".")

    return shifts


def adjust_boxes(boxes, shifts):
    """Shifts the box by the specified amounts in x and y direction.

    This function must be used with the output of the `get_box_shifts` method.
    It shifts the entire box keeping the box dimensions constant.

    Args:
        boxes (`numpy.ndarray`): The K bounding boxes which shall be shifted, as
            array of shape (K, 4) where each row represents the coordinates of a
            single bounding box [xmin, ymin, width, height].

        shifts (`numpy.ndarray`): The desired bounding box shifts in x and y
            direction as returned by the `get_box_shifts` method.

    Returns:
        shifted boxes (`numpy.ndarray`): Same shape as input boxes, but boxes
        are shifted by the specified amounts. If either boxes or shifts are
        an empty array, this functions also returns an empty array of shape
        (0, 4).
    """
    if np.shape(boxes)[0] == 0 or np.shape(shifts)[0] == 0:
        return np.empty(shape=(0, 4))
    else:
        boxes_adjusted = np.copy(boxes)
        boxes_adjusted[:, 0] = boxes[:, 0] + shifts[:, 0]  # xmin + x_shift
        boxes_adjusted[:, 1] = boxes[:, 1] + shifts[:, 1]  # ymin + y_shift
        boxes_adjusted[:, 2] = boxes[:, 2]
        boxes_adjusted[:, 3] = boxes[:, 3]
        return boxes_adjusted


def compute_iou(boxA, boxB):
    """Computes the Intersection over Union (IoU) for two bounding boxes.

    Args:
        boxA, boxB (`numpy.ndarray`): Bounding boxes [xmin, ymin, width, height]
            as arrays with shape (4,) and dtype float.

    Returns:
        IoU (`float`): The IoU of the two boxes. It is within the range [0, 1],
        0 meaning no overlap and 1 meaning full overlap of the two boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs(boxA[2] * boxA[3])
    boxBArea = abs(boxB[2] * boxB[3])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def match_bounding_boxes(t_boxes, d_boxes, iou_threshold):
    """Matches detection boxes with tracked boxes based on IoU.

    This function can be used to find matches between sets of bounding boxes
    found with an object detector and tracked with a tracker. It yields three
    arrays with indices indicating which detected box corresponds to which
    tracked box. This information is needed to update the state of the tracker
    boxes with the correct detection box.

    Matching is performed by the Hungarian Algorithm applied to a cost matrix of
    IoUs of all possible pairs of boxes.

    Args:
        t_boxes (`numpy.ndarray`): Array of shape (T, 4) and dtype float of the
            T tracked bounding boxes in the format [xmin, ymin, width, height]
            each.

        d_boxes (`numpy.ndarray`): Array of shape (D, 4) and dtype float of the
            D detected bounding boxes in the format [xmin, ymin, width, height]
            each.

        iou_threshold (`float`): If the IoU of a detected and a tracked box
            exceeds this threshold they are considered as a match.

    Returns:
        matches (`numpy.ndarray`): Array of shape (M, 2) containing the indices
            of all M matched pairs of detection and tracking boxes. Each row in
            this array has the form [d, t] indicating that the `d`th detection
            box has been matched with the `t`th tracking box (d and t being the
            row indices of d_boxes and t_boxes).

        unmatched_trackers (`numpy.ndarray`): Array of shape (L,) containing
            the L row indices of all tracked boxes which could not be matched to
            any detected box (that is their IoU did not exceed the
            `iou_threshold`). This indicates an event, such as a previously
            tracked target leaving the scene.

        unmatched_detectors (`numpy.ndarray`): Array of shape (K,) containing
            the K row indices of all detected boxes which could not be matched
            to any tracked box. This indicates an event such as a new target
            entering the scene.
    """
    matches = []
    unmatched_trackers = []
    unmatched_detectors = []

    # compute IoU matrix for all possible matches of tracking and detection boxes
    iou_matrix = np.zeros([len(d_boxes), len(t_boxes)])
    for d, d_box in enumerate(d_boxes):
        for t, t_box in enumerate(t_boxes):
            iou_matrix[d, t] = compute_iou(d_box, t_box)
    # find matches between detection and tracking boxes that lead to maximum total IoU
    d_idx, t_idx = scipy.optimize.linear_sum_assignment(-iou_matrix)
    # find all detection boxes, which have no tracker yet
    unmatched_detectors = []
    for d in range(len(d_boxes)):
        if d not in d_idx:
            unmatched_detectors.append(d)
    # find all tracker boxes, which have not been detected anymore
    unmatched_trackers = []
    for t in range(len(t_boxes)):
        if t not in t_idx:
            unmatched_trackers.append(t)
    # filter out matches with low Iou
    matches = []
    for d, t in zip(d_idx, t_idx):
        if iou_matrix[d, t] < iou_threshold:
            unmatched_detectors.append(d)
            unmatched_trackers.append(t)
        else:
            matches.append([d, t])
    if len(matches) == 0:
        matches = np.empty((0, 2))
    else:
        matches = np.vstack(matches)

    # sort descending for later deletion
    unmatched_trackers = np.array(sorted(unmatched_trackers, reverse=True))
    unmatched_detectors = np.array(sorted(unmatched_detectors, reverse=True))
    return matches, unmatched_trackers, unmatched_detectors


class Kalman():
    """Kalman filter for estimation of bounding boxes given noisy measurements.

    Immplements a kalman filter for 8D state space
    xs = [cx, cy, a, h, vx, vy, va, vh] with center position (x, y),
    aspect ratio a and height h of bounding box. The first four parameters
    are directly measured by detection in irregular intervals.
    """
    def __init__(self):
        self.kalman_filter = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.
        # state transition matrix
        self.kalman_filter.F = np.array([[1,0,0,0,dt,0,0,0],
                                         [0,1,0,0,0,dt,0,0],
                                         [0,0,1,0,0,0,dt,0],
                                         [0,0,0,1,0,0,0,dt],
                                         [0,0,0,0,1,0,0,0],
                                         [0,0,0,0,0,1,0,0],
                                         [0,0,0,0,0,0,1,0],
                                         [0,0,0,0,0,0,0,1]], dtype=np.float32)
        # measurement matrix
        self.kalman_filter.H = np.array([[1,0,0,0,0,0,0,0],
                                         [0,1,0,0,0,0,0,0],
                                         [0,0,1,0,0,0,0,0],
                                         [0,0,0,1,0,0,0,0]], dtype=np.float32)
        # measurement noise matrix
        self.kalman_filter.R = np.zeros((4, 4), dtype=np.float32)
        # process noise matrix
        self.kalman_filter.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=1.0, block_size=4, order_by_dim=False)
        #self.kalman_filter.Q = Q_continuous_white_noise(dim=2, dt=1.0, spectral_density=1.0, block_size=4, order_by_dim=False)
        # initial covariance estimate (initially identiy matrix)
        self.kalman_filter.P *= 1000.


    def update(self, box):
        """Update the filter state based on a measured bounding box."""
        box_measured = self._box_to_measurement(box)
        self.kalman_filter.update(box_measured)


    def predict(self):
        """Predict the next filter state."""
        self.kalman_filter.predict()


    def get_box_from_state(self):
        return self._state_to_box(self.kalman_filter.x)


    def set_initial_state(self, box):
        box_measured = self._box_to_measurement(box)
        self.kalman_filter.x = np.append(box_measured, [0., 0., 0., 0.]).reshape(8, 1)


    def _state_to_box(self, state):
        """Converts the filter's state to bounding box coordinates."""
        cx = state[0]
        cy = state[1]
        a = state[2]
        h = state[3]
        w = a*h
        x = cx-w/2
        y = cy-h/2
        return [x, y, w, h]


    def _box_to_measurement(self, box):
        """Converts bounding box coordinates into a measurement vector."""
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cx = x+w/2
        cy = y+h/2
        if h > 0:
            a = w/h
        else:
            a = 0
        box_measured = np.array([cx, cy, a, h], dtype=np.float32).reshape(4, 1)
        return box_measured
