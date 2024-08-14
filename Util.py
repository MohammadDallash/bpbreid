import os
import shutil
import cv2
from ennv import *

def clear_or_create_folder(folder_path):
    """
    Clears all contents of a folder if it exists; otherwise, creates the folder.

    Parameters:
    folder_path (str): The path to the folder to be cleared or created.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        # Remove everything in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory and its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


def get_frame_from_video(frame_num):
    frame_path = os.path.join(temp_frames_folder, f'frame_{frame_num}.jpg')

    frame = cv2.imread(frame_path)
    
    return frame

def is_in_area(center, area):
    x1, y1, x2, y2 = area
    cx, cy = center
    return x1 <= cx <= x2 and y1 <= cy <= y2


def save_tamp_frame_on_disk(frame, frame_index):
     # Ensure that the frame is not empty before saving
    if frame is not None and not frame.size == 0:
        try:
            cv2.imwrite(os.path.join(temp_frames_folder, f'frame_{frame_index}.jpg'), frame)
            return True
        except cv2.error as e:
            print(f"Error saving frame {frame_index}: {e}")
            return False
    else:
        print(f"Warning: Frame {frame_index} is empty or invalid. Skipping.")
        return False


def process_detections(frame, frame_index, tracked_detections, frame_counts, tracking_data):
    for detection, obj_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
        if obj_id not in frame_counts:
            frame_counts[obj_id] = 0
            tracking_data[obj_id] = []
        
        frame_counts[obj_id] += 1

        x1, y1, x2, y2 = map(int, detection)
        tracking_data[obj_id].append((frame_index, (x1, y1, x2, y2)))  # Store frame number and detection box

        # Draw the bounding box and tracker ID on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, frame_counts, tracking_data
        
        


def convert_to_bounding_box(corner_points):
    min_x = min(corner_points[0][0], corner_points[1][0])
    min_y = min(corner_points[0][1], corner_points[1][1])
    max_x = max(corner_points[0][0], corner_points[1][0])
    max_y = max(corner_points[0][1], corner_points[1][1])

    return min_x, min_y, max_x, max_y