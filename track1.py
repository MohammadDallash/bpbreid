from ultralytics import YOLO
import supervision as sv
import torch
import cv2
from tqdm import tqdm
import os
import math
from Util import *
from ennv import *

clear_or_create_folder(temp_frames_folder)



if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("GPU is not available. Using CPU.")

# Build a YOLOv9c model from pretrained weight
yolo_model = YOLO("pretrained_models/yolov8_people.pt")
tracker = sv.ByteTrack()

yolo_model.to(device)

Q = 0

start_sec = Q*15*60 + 60 + 160  # Start tracking from 1 minute
n_mins = 15  # Track for 15 minutes

end_sec = start_sec + n_mins * 60

VIDEO_PATH = 'inputs/videos/camera1.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)
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

pbar = tqdm(total=frames_limit - start_frame, desc="Data Generation in progress")

# Define area_near and area_far as tuples (x1, y1, x2, y2)
# area_near = (100, 0, 200, 100)  # Example coordinates for area_near
area_far = (0, 500, 1000, 500)  # Example coordinates for area_far

while cap.isOpened() and frame_index < frames_limit:
    ret, frame = cap.read()
    if not ret:
        break 

    if not save_tamp_frame_on_disk(frame, frame_index):
        continue


    results = yolo_model(frame, classes=[0], conf=0.35, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(results)
    tracked_detections = tracker.update_with_detections(detections)

    frame, frame_counts, tracking_data = process_detections(frame, frame_index, tracked_detections, frame_counts, tracking_data)

    # Draw the area_far rectangle in blue
    cv2.rectangle(frame, (area_far[0], area_far[1]), (area_far[2], area_far[3]), (255, 0, 0), 2)


    # Draw the frame number on the frame
    cv2.putText(frame, f'Frame: {frame_index}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    # Display the frame with the tracking
    cv2.imshow('Tracking', frame)
    
    frame_index += 1
    pbar.update(1)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pbar.close()
cap.release()
cv2.destroyAllWindows()

output_folder = 'inputs/reid_inputs/gallery'
clear_or_create_folder(output_folder)



for obj_id, frames in tracking_data.items():
    # Get the center point of the first and last occurrences
    first_frame, first_box = frames[0]
    last_frame, last_box = frames[-1]

    # Calculate the center points of the first and last occurrences
    first_center = ((first_box[0] + first_box[2]) // 2, (first_box[1] + first_box[3]) // 2)
    last_center = ((last_box[0] + last_box[2]) // 2, (last_box[1] + last_box[3]) // 2)


    filtered_frames = []

    for frame in frames:
        frame_center = (frame[1][0] + frame[1][2]) // 2, (frame[1][1] + frame[1][3]) // 2

        if  frame[1][3]> 500:
            continue

        filtered_frames.append(frame)



    if filtered_frames == []:
        continue

    selected_frames = [max(filtered_frames, key=lambda f: (f[1][2] - f[1][0]) * (f[1][3] - f[1][1]))]
    

    for i, (frame_num, (x1, y1, x2, y2)) in enumerate(selected_frames):
        frame = get_frame_from_video(frame_num)

   

        cropped_img = frame[y1:y2, x1:x2]


        cv2.imwrite(os.path.join(output_folder, f'{obj_id}_frame_{frame_num}_{i+1}.jpg'), cropped_img)

