# Homework 4: YOLO Object Detection

## Overview

For this assignment, I used YOLOv7 to perform object detection on a sports video and real-time webcam input. The goal was to detect players, detect the sports ball, and test the model on webcam input to find a clear mis-detection.

I used a pretrained YOLOv7 model, so I did not train a new model from scratch. The detections were based on the COCO pretrained object classes.

## Files Submitted

This submission includes the following files:

1. `task1_players_detection.mp4`  
   Resulting video for Task 1, where YOLO detects the players in the video.

2. `task2_sports_ball_detection.mp4`  
   Resulting video for Task 2, where YOLO detects the sports ball in the video.

3. `task3_webcam_misdetection.png`  
   Screenshot from the webcam task showing an object that the model mis-detected.

4. `commands.txt`  
   A record of the command lines I used and the related terminal outputs.

5. `README.md`  
   Explanation of the commands, problems found in the results, possible reasons, potential improvements, and my understanding of YOLO.

## Model and Repository Used

I followed the YOLOv7 object tracking repository provided in the assignment:

```bash
https://github.com/RizwanMunawar/yolov7-object-tracking
````

The model used pretrained YOLOv7 weights. Since the assignment focuses on applying YOLO for object detection, I used the pretrained model directly instead of training a custom model.

## Task 1: Player Detection

For Task 1, I used YOLO to detect players in the sports video. In the COCO dataset, players are detected under the general class `person`, so I filtered the detection results to class `0`.

Command used:

```bash
python detect_and_track.py --weights yolov7.pt --source input_30s.mp4 --classes 0 --conf-thres 0.25 --name task1_players --exist-ok
```

### Command Explanation

* `python detect_and_track.py` runs the YOLOv7 detection and tracking script.
* `--weights yolov7.pt` loads the pretrained YOLOv7 weights.
* `--source input_30s.mp4` uses my selected video clip as the input source.
* `--classes 0` filters the detections to the COCO class `person`.
* `--conf-thres 0.25` keeps detections with confidence scores above 0.25.
* `--name task1_players` saves the result into a folder named for Task 1.
* `--exist-ok` allows the output folder name to be reused if it already exists.

The output video was saved and renamed as:

```text
task1_players_detection.mp4
```

## Task 2: Sports Ball Detection

For Task 2, I used YOLO to detect the sports ball in the same video clip. In the COCO dataset, the class index for `sports ball` is `32`.

Command used:

```bash
python detect_and_track.py --weights yolov7.pt --source input_30s.mp4 --classes 32 --conf-thres 0.10 --name task2_ball --exist-ok
```

### Command Explanation

* `--classes 32` filters the detections to the COCO class `sports ball`.
* `--conf-thres 0.10` uses a lower confidence threshold because the ball is small and harder to detect.
* The rest of the command uses the same pretrained YOLOv7 model and input video.

The output video was saved and renamed as:

```text
task2_sports_ball_detection.mp4
```

## Task 3: Real-Time Webcam Detection

For Task 3, I used the webcam as the input source and tested several real-world objects. I looked for an object that the model clearly mis-detected and saved a screenshot.

Command used:

```bash
python detect_and_track.py --weights yolov7.pt --source 0 --conf-thres 0.25 --view-img --name task3_webcam --exist-ok
```

### Command Explanation

* `--source 0` uses the default webcam as the input source.
* `--view-img` displays the webcam detection window in real time.
* `--conf-thres 0.25` keeps detections above the confidence threshold.
* I captured a screenshot when the model made a clear mis-detection.

The screenshot was saved as:

```text
task3_webcam_misdetection.png
```

## Problems Found in the Results

The player detection result was generally better than the sports ball detection result. Players are larger in the frame and have more recognizable visual features, so YOLO was able to detect them more consistently.

The sports ball detection was less stable. In some frames, the ball was missed, and in other frames the confidence score was low. This likely happened because the ball was small, moving quickly, partially occluded, or visually blended into the background.

For the webcam task, the model mis-detected one object because the pretrained YOLOv7 model can only classify objects into the categories it learned from the COCO dataset. If the object is not included in those categories, or if it appears from an unusual angle, the model may assign an incorrect but visually similar label.

## Reasons for the Problems

Several factors may have caused the detection errors:

1. Small object size
   The sports ball occupies only a small region of the frame, making it harder for the model to detect.

2. Motion blur
   Since the ball moves quickly, it can appear blurred in the video.

3. Occlusion
   Players may block the ball, especially during active moments in the video.

4. Background similarity
   The ball can blend into the field, players, or other objects in the scene.

5. Limited pretrained categories
   The webcam mis-detection happened partly because the pretrained model only recognizes predefined COCO classes.

6. No custom training
   I used a pretrained model rather than fine-tuning it on this specific sports video or on my webcam objects.

## Potential Improvements

The results could be improved in several ways:

1. Use higher-resolution video input so that small objects like the ball are clearer.
2. Adjust the confidence threshold depending on the task.
3. Fine-tune YOLO on a sports-specific dataset with more examples of players and balls.
4. Use a newer or more accurate object detection model.
5. Use tracking methods to maintain ball detection across frames even when the ball is temporarily missed.
6. Improve lighting and camera angle for webcam detection.
7. Add more training examples for objects that are commonly mis-detected.

## My Understanding of YOLO

YOLO stands for “You Only Look Once.” It is a one-stage object detection model that predicts bounding boxes and class probabilities directly from an image in a single forward pass.

Compared with two-stage detectors, YOLO is usually faster because it does not first generate many region proposals. This makes YOLO useful for video detection and real-time webcam detection.

A pretrained YOLO model can detect common object categories without additional training, but it has limitations. It works best when the objects are similar to the examples it saw during training. It may perform worse on small objects, blurry objects, occluded objects, unusual camera angles, or objects outside its training categories.

In this assignment, YOLO worked well for detecting players because they are large and belong to the common `person` class. It had more difficulty detecting the sports ball because the ball was much smaller and moved quickly.

