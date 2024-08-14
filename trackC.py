from ultralytics import YOLO
from ultralytics import RTDETR

import supervision as sv
import torch
import cv2
from tqdm import tqdm
import os
import math
from deep_sort_realtime.deepsort_tracker import DeepSort
from Util import *
import numpy as np
from time import time


# Initialize global variables for rectangles and dragging
dragging_near = False
dragging_far = False
selected_corner = None

# Define the scaling factor
scaling_factor = 0.5

# Scaled areas for display purposes
area_far = [(100, 50), (200, 150)]  # Top-left and bottom-right corners
area_near = [(200, 300), (800, 450)]  # Top-left and bottom-right corners


def scale_point(point, factor):
    return int(point[0] * factor), int(point[1] * factor)

def unscale_point(point, factor):
    return int(point[0] / factor), int(point[1] / factor)

def mouse_callback(event, x, y, flags, param):
    global dragging_near, dragging_far, selected_corner

    # Unscale the mouse coordinates to match the original frame size
    x_unscaled, y_unscaled = unscale_point((x, y), scaling_factor)

    if event == cv2.EVENT_LBUTTONDOWN:
        if abs(x_unscaled - area_far[0][0]) < 10 and abs(y_unscaled - area_far[0][1]) < 10:
            dragging_near = True
            selected_corner = 0
        elif abs(x_unscaled - area_far[1][0]) < 10 and abs(y_unscaled - area_far[1][1]) < 10:
            dragging_near = True
            selected_corner = 1
        elif abs(x_unscaled - area_near[0][0]) < 10 and abs(y_unscaled - area_near[0][1]) < 10:
            dragging_far = True
            selected_corner = 0
        elif abs(x_unscaled - area_near[1][0]) < 10 and abs(y_unscaled - area_near[1][1]) < 10:
            dragging_far = True
            selected_corner = 1

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_near:
            area_far[selected_corner] = (x_unscaled, y_unscaled)
        elif dragging_far:
            area_near[selected_corner] = (x_unscaled, y_unscaled)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_near = False
        dragging_far = False
        selected_corner = None


clear_or_create_folder(temp_frames_folder)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("GPU is not available. Using CPU.")

# Load a COCO-pretrained RT-DETR-l model
yolo_model = RTDETR("pretrained_models/rtdetr-l.pt")
yolo_model.to(device)

Q = 0
start_sec = Q * 15 * 60 + 60  # Start tracking from 1 minute
n_mins = 15
end_sec = start_sec + n_mins * 60

cap = cv2.VideoCapture(VID_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS of the video: {fps}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

start_frame = math.floor(start_sec * fps)
end_frame = math.floor(end_sec * fps)
number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
tracking_data = {}
frame_counts = {}
frame_index = start_frame
frames_limit = min(end_frame, number_of_frames)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Set the mouse callback
cv2.namedWindow('Adjust Areas')
cv2.setMouseCallback('Adjust Areas', mouse_callback)

ret, frame = cap.read()

# Initialize Deep SORT
tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.2, embedder_gpu=True)

while True:
    if not ret:
        break

    # Resize the frame for display
    scaled_frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)

    # Scale areas for display
    scaled_near = [scale_point(pt, scaling_factor) for pt in area_far]
    scaled_far = [scale_point(pt, scaling_factor) for pt in area_near]

    # Draw the scaled rectangles
    cv2.rectangle(scaled_frame, scaled_near[0], scaled_near[1], (0, 0, 255), 2)
    cv2.rectangle(scaled_frame, scaled_far[0], scaled_far[1], (255, 0, 0), 2)

    cv2.imshow('Adjust Areas', scaled_frame)

    key = cv2.waitKey(1)
    if key == ord('\r'):  # Press Enter to start tracking
        break
    elif key == ord('q'):  # Press 'q' to quit
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow('Adjust Areas')

pbar = tqdm(total=frames_limit - start_frame, desc="Data Generation in progress")

area_far = convert_to_bounding_box(area_far)
area_near = convert_to_bounding_box(area_near)

while cap.isOpened() and frame_index < frames_limit:

    ret, frame = cap.read()

    if not ret:
        break 

    if not save_tamp_frame_on_disk(frame, frame_index):
        continue
    
    results = yolo_model(frame, classes=[0], conf=0.10, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Extract the bounding boxes in xyxy format from the detections
    bbox_xyxy = detections.xyxy
    

    # Convert from xyxy to xywh format
    bbox_xywh = []
    for bbox in bbox_xyxy:
        x1, y1, x2, y2 = bbox
        w = x2 - x1         
        h = y2 - y1         
        bbox_xywh.append([x1, y1, w, h])

    # Convert to numpy array for Deep SORT
    bbox_xywh = np.array(bbox_xywh)


    # Confidence scores from detections
    confs = detections.confidence

    classes = [0] * len(confs)
    # Combine the arrays into a list of tuples
    detections = [(bbox.tolist(), conf, cls) for bbox, conf, cls in zip(bbox_xywh, confs, classes)]

    s = time()

    # Update tracker with current frame detections
    outputs = tracker.update_tracks(raw_detections = detections, frame = frame)
    # Convert outputs from Deep SORT to a format compatible with your tracking pipeline

    print(time() - s)
    class TrackedDetections:
        def __init__(self):
            self.xyxy = []
            self.tracker_id = []

    # Instantiate the class
    tracked_detections = TrackedDetections()

    # Populate the class with bounding boxes and IDs from Deep SORT outputs
    for track in outputs:
        bbox = track.to_tlbr()  # Convert to (x1, y1, x2, y2)
        tracked_detections.xyxy.append(bbox)
        tracked_detections.tracker_id.append(track.track_id)



    frame, frame_counts, tracking_data = process_detections(frame, frame_index, tracked_detections, frame_counts, tracking_data)

    # # Draw the area_far rectangle in red
    # cv2.rectangle(frame, (area_far[0], area_far[1]), (area_far[2], area_far[3]), (0, 0, 255), 2)

    # # Draw the area_near rectangle in blue
    # cv2.rectangle(frame, (area_near[0], area_near[1]), (area_near[2], area_near[3]), (255, 0, 0), 2)

    # # Draw the frame number on the frame
    # cv2.putText(frame, f'Frame: {frame_index}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # # Display the frame with the tracking
    # cv2.imshow('Tracking', frame)
    
    # frame_index += 1
    pbar.update(1)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pbar.close()
cap.release()
cv2.destroyAllWindows()

query_output_folder = 'inputs/reid_inputs/query'
clear_or_create_folder(query_output_folder)

gallery_output_folder = 'inputs/reid_inputs/gallery'
clear_or_create_folder(gallery_output_folder)

def propose_best(frames, area):
    filtered_frames = []

    for frame in frames:
        frame_center = (frame[1][0] + frame[1][2]) // 2, (frame[1][1] + frame[1][3]) // 2

        if is_in_area(frame_center, area):
            filtered_frames.append(frame)

    if not filtered_frames:
        return None

    biggest_frame = max(filtered_frames, key=lambda f: (f[1][2] - f[1][0]) * (f[1][3] - f[1][1]))

    return biggest_frame

def save_img(frame, output_folder):
    frame_num, (x1, y1, x2, y2) = frame

    try:
        frame = get_frame_from_video(frame_num)

        cropped_img = frame[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(output_folder, f'{obj_id}.jpg'), cropped_img)
    except Exception as e:
        print(f"Error processing frame {frame_num}: {e}")

for obj_id, frames in tracking_data.items():

    biggest_frame_near = propose_best(frames, area_near)
    biggest_frame_far = propose_best(frames, area_far)

    if biggest_frame_near:
        save_img(biggest_frame_near, query_output_folder)

    if biggest_frame_far:
        save_img(biggest_frame_far, gallery_output_folder)
