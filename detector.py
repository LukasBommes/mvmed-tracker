import os
import logging
import json
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import cv2


#################################################################################
#
#       OBJECT DETECTOR BASE CLASS
#
#################################################################################


class _BaseDetector(ABC):
    """Abstract interface for different types of object detectors.

    This class serves as a template for the implmentation of specific object
    detectors. To create a specific detector, it has to subclass this interface
    and provide an implementation for the method :py::meth:`._get_output_dict`.

    Args:
        path (`str`): Base directory of the object detector model. This is
            needed to load the `labelmap.json` file which contains the mapping
            from integer class labels to label strings.

        box_size_threshold (`tuple` of `float` or `None`): Specifies the maximum
            allowed width and height of detected bounding boxes. Larger boxes
            are filtered out. This is useful if the detector yields unnaturally
            large bounding boxes which do not represent any true object. If
            `None` no such filtering is applied.

        scaling_factor (`float` or `None`): Bounding boxes in the detector
            output are resized by this factor to match potentially resized
            video frames. E.g. if video frames are resized by a factor of 0.5
            the bounding box coordinates have to be multipled by the same factor
            to match the object silhouttes in the resized frames. If `None` no
            rescaling is applied.
    """
    def __init__(self, path, box_size_threshold, scaling_factor):
        self.path = path
        self.box_size_threshold = box_size_threshold
        self.scaling_factor = scaling_factor
        self._load_label_map()


    def _load_label_map(self):
        """Load the map for class labels from labelmap.json file."""
        self.label_map = []
        model_dir = os.path.dirname(self.path)
        label_map_file = os.path.join(model_dir, "labelmap.json")
        if os.path.isfile(label_map_file):
            self.label_map = json.load(open(label_map_file, "r"))


    def _class_index_to_label(self, class_index):
        """Maps a class index to a class string.

        Args:
            class_index (`int`): A predicted numeric class index. This index
                needs to have a corresponing class string in the loaded
                `labelmap.json` file.

        Returns:
            (`str`): The class string which corresponds to the provided class
            index.

        Raises:
            RuntimeError: When the label map is empty, that is if no labelmap
                was loaded previously.
        """
        if len(self.label_map) == 0:
            raise RuntimeError(("No label map was loaded. Either this model "
                "does not support label mapping or the label map file could "
                "not be loaded."))
        else:
            class_label = [map["label"] for map in self.label_map if map["index"] == class_index][0]
        return class_label


    @abstractmethod
    def _get_output_dict(self, frame):
        """Get output dictionary of a detector for a given video frame.

        This is an abstract method and has to be reimplemented by subclasses of
        this interface. It's task is to implement the actual detection process
        on a single video frame. Thus, this method takes a video frame as input
        and returns the detections.

        Args:
            frame (`numpy.ndarray`): A uint8 numpy array of shape `(H x W x C)`
                where `H` stands for the frame height, `W` for the frame width
                and `C` for the number of color channels (for a color image
                C = 3, for gray scale image C = 1).

        Returns:
            (`dict`): Detected objects in the provided frame. The dictionary has
            the following key value pairs:\n
            - 'num_detections' (`int`): The number `N` of detected objects in
              the frame.
            - 'detection_classes' (`numpy.ndarray`): The class label for each
              detected object. The array has shape `(N,)` and contains class
              labels in text form (data type unicode string).
            - 'detection_boxes' (`numpy.ndarray`): The bounding boxes of all `N`
              detected objects. Array has shape `(N, 2)` and data type float32.
              Each row contains bounding box coordinates in the format `(xmin,
              ymin, width, height)` for one of the detected objects. The order
              of rows is consistent with the order of the class labels and
              detections scores.
            - 'detection_scores' (`numpy.ndarray`): Probabilities for every
              detected box to contain an object of the predicted class. Array
              has shape `(N,)` and data type float32. Probabilities are in range
              `[0, 1]`.
        """
        pass


    def _rescale_bounding_boxes(self, output_dict):
        """Rescale bounding box coordinates in-place.

        Args:
            output_dict (`dict`): The detection output as returned by
                :py:meth:`._get_output_dict`.

        Returns:
            (`dict`) The detection output dictionary which was passed as an
            argument with bounding boxes rescaled by the
            :py:attr:`.scaling_factor` provided in the constructor.
        """
        output_dict['detection_boxes'] = np.multiply(output_dict['detection_boxes'], self.scaling_factor)
        return output_dict


    def _filter_oversized_bounding_boxes(self, output_dict):
        """Filters out unnaturally large bounding boxes.

        Args:
            output_dict (`dict`): The detection output as returned by
                :py:meth:`._get_output_dict`.

        Returns:
            (`dict`) A copy of the detection output dictionary which was passed
            as an argument. In this copy all detections with bounding boxes
            larger than the width and height specified in the tuple
            :py:attr:`.box_size_threshold` are removed. According class labels
            and class scores are removed as well.
        """
        new_output_dict = {}
        new_output_dict['num_detections'] = 0
        new_output_dict['detection_boxes'] = []
        new_output_dict['detection_classes'] = []
        new_output_dict['detection_scores'] = []
        # fill entries into new dict, if the box size does not exceed width and height thresholds
        for i, bounding_box in enumerate(output_dict['detection_boxes']):
            if bounding_box[2] < self.box_size_threshold[0] and bounding_box[3] < self.box_size_threshold[1]:
                new_output_dict['detection_boxes'].append(output_dict['detection_boxes'][i])
                new_output_dict['detection_classes'].append(output_dict['detection_classes'][i])
                new_output_dict['detection_scores'].append(output_dict['detection_scores'][i])
                new_output_dict['num_detections'] = new_output_dict['num_detections'] + 1
       # convert to numpy arrays
        new_output_dict['detection_boxes'] = np.array(new_output_dict['detection_boxes'])
        new_output_dict['detection_classes'] = np.array(new_output_dict['detection_classes'])
        new_output_dict['detection_scores'] = np.array(new_output_dict['detection_scores'])
        return new_output_dict


    def detect(self, frame):
        """Run inference in object detection model.

        This function is a wrapper around :py:meth:`._get_output_dict` and takes
        the same input argument and returns the same value. However, it handles
        the case of an invalid input frame (`None`) and applies some
        post-processing to the detection output dictionary. If a
        :py:attr:`.box_size_threshold` was provided in the constructor it
        filters out boxes larger than this. If a :py:attr:`.scaling_factor`
        was specified in the constructor it rescales the bounding box
        coordinates by the same factor.

        Args:
            frame (`numpy.ndarray` or `None`): A uint8 numpy array of shape
                `(H x W x C)` where `H` stands for the frame height, `W` for the
                frame width and `C` for the number of color channels (for a
                color image C = 3, for gray scale image C = 1). If `None`, an
                empty output array is returned.


        Returns:
            (`dict`): Detected objects in the provided frame. The dictionary has
            the following key value pairs:\n
            - 'num_detections' (`int`): The number `N` of detected objects in
              the frame.
            - 'detection_classes' (`numpy.ndarray`): The class label for each
              detected object. The array has shape `(N,)` and contains class
              labels in text form (data type unicode string).
            - 'detection_boxes' (`numpy.ndarray`): The bounding boxes of all `N`
              detected objects. Array has shape `(N, 2)` and data type float32.
              Each row contains bounding box coordinates in the format `(xmin,
              ymin, width, height)` for one of the detected objects. The order
              of rows is consistent with the order of the class labels and
              detections scores.
            - 'detection_scores' (`numpy.ndarray`): Probabilities for every
              detected box to contain an object of the predicted class. Array
              has shape `(N,)` and data type float32. Probabilities are in range
              `[0, 1]`.
              In case the frame is `None` the returned dictionary still contains
              the four described keys. However, the value for 'num_detections'
              is 0 and the arrays are empty. Shape and data type of the empty
              arrays are preserved.
        """
        if frame is not None:
            output_dict =  self._get_output_dict(frame)
            # filter out too large bounding boxes
            if self.box_size_threshold is not None:
                output_dict = self._filter_oversized_bounding_boxes(output_dict)
            # rescale bounding box coordinates according to scaling factor
            if self.scaling_factor is not None:
                output_dict = self._rescale_bounding_boxes(output_dict)
        else:
            output_dict = {
                'num_detections': 0,
                'detection_boxes': np.empty(shape=(0, 4), dtype=np.float32),
                'detection_scores': np.empty(shape=(0,), dtype=np.float32),
                'detection_classes': np.empty(shape=(0,), dtype=np.str_)
            }
        return output_dict


    def draw_bounding_boxes(self, frame, output_dict, color=(0, 255, 0)):
        """Draws detector results on the provided frame.

        Serves as a helper to visualize detection results of :py:meth:`.detect`.
        Bounding boxes and class labels are drawn on the provided frame. For
        meaningful results, this frame should be the same as the one provided
        for inference.

        Args:
            frame (`numpy.ndarray`): A uint8 numpy array of shape `(H x W x C)`
                where `H` stands for the frame height, `W` for the frame width
                and `C` for the number of color channels (for a color image
                C = 3, for gray scale image C = 1). Frame dimensions should be
                the same as the ones of the frame provided during the inference
                in :py:meth:`.detect`. The frame is modified in-place.

            output_dict (`dict`): The detection output as returned by
                :py:meth:`.detect`.

            color (`tuple` of `int`): The color in which to draw the bounding
                boxes. Values are in BGR-order and have to be in range
                `[0, 255]`.
        """
        cv2.putText(frame, 'Detections: {}'.format(output_dict['num_detections']), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        for box, cls, score in zip(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores']):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[0] + box[2])
            ymax = int(box[1] + box[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, '{}'.format(cls), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, '{:1.3f}'.format(score), (xmin, ymax-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


#################################################################################
#
#       OBJECT DETECTOR OF TENSORFLOW OBJECT DETECTION API
#
#################################################################################


class DetectorTF(_BaseDetector):
    """Interface for object detectors of the Tensorflow Object Detection API.
    GitHub: [https://github.com/tensorflow/models/tree/master/research/object_detection]
    """

    def __init__(self, path, box_size_threshold=None, scaling_factor=None, gpu=0):
        """Setup the object detetor.
        This methods initializes the object detector. The path argument specifies the location
        of the frozen graph file (*.pb) on the hard disk. The gpu argument specifies the id (int)
        of the GPU on which to run the object detector. The function returns a detector object.
        """
        super().__init__(path, box_size_threshold, scaling_factor)
        self.logger = logging.getLogger(__name__)
        # Load frozen tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                with tf.device('/gpu:{}'.format(gpu)):
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            # Get tensors from graph
            tfconfig = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False)
            tfconfig.gpu_options.allow_growth=True
            self.sess = tf.Session(config=tfconfig)
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            self.logger.info(("DetectorTF (ID {}): Initialized object detector "
                "(model path: {}, box size threshold: {}, scaling factor: {}, "
                "gpu: {}).").format(id(self), path, box_size_threshold,
                scaling_factor, gpu))


    def _get_output_dict(self, frame):
        """Get output dictionary of detector for a given frame."""
        # Run inference
        output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: np.expand_dims(frame, 0)})
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        n_detections = output_dict['num_detections']
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0][:n_detections]
        output_dict['detection_scores'] = output_dict['detection_scores'][0][:n_detections]
        output_dict['detection_classes'] = output_dict['detection_classes'][0][:n_detections].astype(np.uint8)
        # convert class indices to labels
        cls_labels = []
        for cls_idx in output_dict['detection_classes']:
            cls_labels.append(self._class_index_to_label(cls_idx))
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.str_)
        output_dict['detection_classes'] = cls_labels
        # convert format of bounding boxes to (xmin, ymin, width, height)
        frame_width = np.shape(frame)[1]
        frame_height = np.shape(frame)[0]
        output_dict = self._convert_bounding_box_format(output_dict, frame_width, frame_height)
        return output_dict


    def _convert_bounding_box_format(self, output_dict, frame_width, frame_height):
        """Converts format of all bounding boxes in output_dict.
        Method modifies the bounding box format of all boxes in the detector's output_dict.
        The new format of each box (a row in output_dict['detection_boxes']) is (xmin, ymin, width, heigth).
        Box coordinates are determined based on the size of the frame fed into detect method.
        """
        output_dict['detection_boxes'][:, 0] = np.multiply(output_dict['detection_boxes'][:, 0], frame_height)  # ymin
        output_dict['detection_boxes'][:, 1] = np.multiply(output_dict['detection_boxes'][:, 1], frame_width)  # xmin
        output_dict['detection_boxes'][:, 2] = np.multiply(output_dict['detection_boxes'][:, 2], frame_height)  # ymax
        output_dict['detection_boxes'][:, 3] = np.multiply(output_dict['detection_boxes'][:, 3], frame_width)  # xmax
        new_output_dict = output_dict.copy()
        new_output_dict['detection_boxes'] = np.copy(output_dict['detection_boxes'])
        new_output_dict['detection_boxes'][:, 0] = output_dict['detection_boxes'][:, 1]  # xmin
        new_output_dict['detection_boxes'][:, 1] = output_dict['detection_boxes'][:, 0]  # ymin
        new_output_dict['detection_boxes'][:, 2] = output_dict['detection_boxes'][:, 3] - output_dict['detection_boxes'][:, 1]  # width
        new_output_dict['detection_boxes'][:, 3] = output_dict['detection_boxes'][:, 2] - output_dict['detection_boxes'][:, 0]  # height
        return new_output_dict
