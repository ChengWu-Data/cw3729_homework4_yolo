# Homework 4: YOLO Object Detection

## Overview

For this assignment, I used YOLOv7 to perform object detection on a soccer match video and on live webcam input. The main goal was to test how a pretrained object detection model performs in different settings.

I completed three tasks:

1. Detect the players in a sports video  
2. Detect the sports ball in the same video  
3. Use webcam detection and capture a clear example of mis-detection

Instead of training a new model, I used the pretrained YOLOv7 weights provided by the repository.

---

## Files Included

- `task1_players_detection.mp4`
- `task2_sports_ball_detection.mp4`
- `task3_webcam_misdetection.png`
- `commands.txt`
- `README.md`

---

## Model Used

I used the YOLOv7 object tracking repository provided in the assignment:

https://github.com/RizwanMunawar/yolov7-object-tracking

YOLOv7 was already trained on the COCO dataset, so it can recognize common categories such as person, sports ball, car, chair, bottle, and many others.

---

## Task 1: Detecting Players

Command used:

```bash
python detect_and_track.py --weights yolov7.pt --source ../football.mp4 --classes 0 --conf-thres 0.25 --name task1_players --exist-ok
````

I used class `0`, which corresponds to `person` in the COCO dataset.

Since players are relatively large and clearly visible in most frames, the detection result was generally strong. Most players were identified correctly, even while moving.

---

## Task 2: Detecting the Sports Ball

Command used:

```bash
python detect_and_track.py --weights yolov7.pt --source ../football.mp4 --classes 32 --conf-thres 0.10 --name task2_ball --exist-ok
```

I used class `32`, which corresponds to `sports ball`.

Compared with player detection, ball detection was more difficult. The ball is much smaller than the players and moves quickly. In several frames, the model missed the ball completely or detected it only briefly.

Reducing the confidence threshold helped produce more detections.

---

## Task 3: Webcam Mis-detection

Command used:

```bash
python detect_and_track.py --weights yolov7.pt --source 0 --conf-thres 0.25 --view-img --name task3_webcam --exist-ok
```

I tested several everyday objects using my webcam and captured a screenshot when the model clearly assigned the wrong label.

This demonstrates an important limitation of pretrained models: they only predict categories they already know, so unfamiliar objects are often labeled as the closest known class.

---

## Problems Observed

The model performed better on players than on the ball.

Main issues I noticed:

* Small objects were harder to detect
* Fast motion reduced accuracy
* Some frames missed the ball entirely
* Occlusion from players sometimes blocked the ball
* Webcam objects were occasionally mislabeled

These problems are common in real-time object detection.

---

## Possible Improvements

There are several ways the results could be improved:

* Use higher resolution video
* Fine-tune the model on soccer-specific data
* Use a stronger or newer YOLO version
* Apply object tracking to recover missed detections
* Improve camera angle and lighting for webcam input

---

## My Understanding of YOLO

YOLO stands for “You Only Look Once.”

Unlike older two-stage detectors, YOLO predicts object locations and labels in a single pass through the image. This makes it fast enough for video and real-time webcam use.

The main advantage of YOLO is speed with strong overall accuracy. However, performance usually drops on very small objects, blurred motion, crowded scenes, or categories outside the training dataset.

This assignment showed that clearly: player detection was reliable, while sports ball detection was much more challenging.
