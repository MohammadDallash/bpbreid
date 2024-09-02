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

  


