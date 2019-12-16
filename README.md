# MVmed: Fast Multi-Object Tracking in the Compressed Domain

This is the code base of the paper "MVmed: Fast Multi-Object Tracking in the Compressed Domain".

MVmed is a real-time online tracker for pedestrians and objects in MPEG-4 and H.264 compressed videos. Tracking is performed based on motion vectors extracted from the compressed video stream. Extraction is performed with our mv-extractor tool which can be found [here](https://github.com/LukasBommes/mv-extractor).

The image below shows an example output of a sequence taken from the [MOT Challenge](https://motchallenge.net/) dataset.

![tracker_output_image](tracker_output.png)

The code contains our MVmed tracker and a Faster R-CNN object detector. The detector is run in regular intervals and provides detections which are matched to tracked bounding boxes. Matching is performed by means of the Hungarian algorithm so as to maximize the overall IoU of all boxes. For intermediate frames MVmed averages motion vectors in each bounding box in the previous frame to predict the corresponding bounding box in the current frame. Novel is that MVmed does not require the object detector to be run exclusively on key frames. Instead detection and prediction are scheduled regardless of the frame type. In case prediction is to be performed on a key frame (which has no motion vectors), motion vector from the previous frame are reused. MVmed can be configured to use both P and B frames or P frames alone.

## Installation

Install [Docker](https://docs.docker.com/).
Clone the repo to your machine
```
git clone -b "v1.0.0" https://github.com/LukasBommes/mvmed-tracker.git mvmed_tracker
```
Open a terminal inside the repo and build the docker container with the following command (note: this can take several hours)
```
sudo docker build . --tag=mvmed_tracker
```
Now, run the docker container with
```
sudo docker run -it --ipc=host --env="DISPLAY" -v $(pwd):/mvmed_tracker -v /tmp/.X11-unix:/tmp/.X11-unix:rw mvmed_tracker /bin/bash
```
Test if everything is succesfully installed by running the demo script
```
python3 track.py
```

## Usage

For a usage example please see the `track.py` script.

#### Preparing the Input Video

The tracker requires an input video that is encoded with MPEG-4 or H.264 codec. To create a suitable input video, [FFMPEG](https://ffmpeg.org/) can be used.

To convert a set of individual RGB images into a suitable video for tracking use the following command from within the directory containing the images:
```
ffmpeg -y -r <f> -i %06d.jpg -c:v mpeg4 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -f rawvideo out.mp4
```
Here, `<f>` is the desired frame rate of the output video.

#### Explanation of the Tracker Parameters

You might want to change the parameters in this script to see how they affect tracking performance. Below is a short explanation of each important parameter:

- **codec (str)**: "mpeg4" or "h264". Used only to select whether to use the MPEG-4 or H.264 encoded example video.
- **video_file (str)**: The video file to used for tracking.
- **use_offline_detections (bool)**: If set to True the test script will use the MOT detections provided in `data/det.txt` instead of computing detections online with a Faster R-CNN detector.
- **detections_file (str)**: The path to the offline detection file (`det.txt`)
- **num_frames (int)**: Number of video frames in the input video (only needed when offline detections are used).
- **detector_path (str)**: Path to the frozen inference graph (`*.pb`) of the object detector.
- **detector_box_size_thres (tuple of 2 floats)**: Provide two values (width, height). Detection boxes larger than this will be filtered out. Set to None to disable filtering. Useful if you have a static cam and targets are known not to exceed a certain size. Settings this filter can then help to prevent oversized false positive detections.
- **detector_interval (int)**: How often the detector is run. E.g. a value of 12 means the detector is run on every 12th frame.
- **tracker_iou_thres (float)**: Discard matches between detections and tracked boxes during data association with an IoU below this threshold value.
- **det_conf_threshold (float)**: Discard detections with a confidence score below this value.
- **state_thresholds (tuple of 3 ints)**: Thresholds `(p->a, a->p, p->d)` for the state machine of tracked targets. `p->a` is the number of consecutive detections needed before a target is set from pending to active state. Only active targets are considered in the tracker output. `a->p` is the maximum number of misses of a tracked target before it is set from active to pending state. `p->d` determines how many misses are allowed before a pending target is deleted permanently.
- **step_wise (bool)**: If True stop after each frame. Key 's' needs to be pressed to advance to the next frame.

#### Changing the Object Detector

The tracker is compatible with all objects detector provided in the Tensorflow [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Just download the detector you want to use from the model zoo provided in the TF repository and place it anywhere. Update the `detector_path` in the `track.py` script to point to the frozen inference graph file `*.pb` of the downloaded detector.

## Citation

If you find our work useful please cite
´´´

´´´
