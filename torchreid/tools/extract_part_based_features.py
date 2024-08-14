import torch
import tqdm
import glob
import time
import os
import numpy as np
from torchreid.scripts.default_config import get_default_config, display_config_diff
from torchreid.tools.feature_extractor import FeatureExtractor
from torchreid.utils.constants import *
from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
import cv2
from Util import clear_or_create_folder
import matplotlib.pyplot as plt
from ennv import *
import json

test_embeddings = ['bn_foreg', 'parts']

def extract_test_embeddings(model_output):
        embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, parts_masks = model_output
        embeddings_list = []
        visibility_scores_list = []
        embeddings_masks_list = []

        for test_emb in test_embeddings:
            embds = embeddings[test_emb]
            embeddings_list.append(embds if len(embds.shape) == 3 else embds.unsqueeze(1))
            if test_emb in bn_correspondants:
                test_emb = bn_correspondants[test_emb]
            vis_scores = visibility_scores[test_emb]
            visibility_scores_list.append(vis_scores if len(vis_scores.shape) == 2 else vis_scores.unsqueeze(1))
            pt_masks = parts_masks[test_emb]
            embeddings_masks_list.append(pt_masks if len(pt_masks.shape) == 4 else pt_masks.unsqueeze(1))

        assert len(embeddings) != 0

        embeddings = torch.cat(embeddings_list, dim=1)  # [N, P+2, D]
        visibility_scores = torch.cat(visibility_scores_list, dim=1)  # [N, P+2]
        embeddings_masks = torch.cat(embeddings_masks_list, dim=1)  # [N, P+2, Hf, Wf]

        return embeddings, visibility_scores, embeddings_masks, pixels_cls_scores


def list_images_in_folder(folder_path):
    # Define the file types you want to list
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']

    # List to hold the file paths
    image_files = []

    # Loop through the file types and add them to the image_files list
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))

    return image_files

def extract_part_based_features(extractor, image_list, store_in_gpu = False,batch_size=400):

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    all_embeddings = []
    all_visibility_scores = []

    images_chunks = chunks(image_list, batch_size)

    for chunk in tqdm.tqdm(images_chunks, desc=f'Batches processed '):
        model_out = extractor(chunk)

        embeddings, visibility_scores, masks, pixels_cls_scores = extract_test_embeddings(model_out)

        if not store_in_gpu:
            embeddings = embeddings.cpu().detach()
            visibility_scores = visibility_scores.cpu().detach()

        all_embeddings.append(embeddings)
        all_visibility_scores.append(visibility_scores)

    
    all_embeddings = torch.cat(all_embeddings, 0)
    all_visibility_scores = torch.cat(all_visibility_scores, 0)


    return  all_embeddings, all_visibility_scores
    

def extract_det_idx(img_path):
    return int(os.path.basename(img_path).split("_")[0])


def get_top_k_closest_gallery_images(query_idx, distmat, gallery_images, k=5):
    distances = distmat[query_idx]
    top_k_indices = distances.argsort()[:k]
    top_k_gallery_images = [gallery_images[idx] for idx in top_k_indices]
    top_k_distances = [distances[idx].item() for idx in top_k_indices]
    return top_k_gallery_images, top_k_distances


import os
import cv2
import matplotlib.pyplot as plt

def save_images(query_idx, query_images, top_k_gallery_images, top_k_distances, output_folder, plt_imgs=False):
    # Load the query image
    query_image = cv2.imread(query_images[query_idx])
    query_image_file_name = os.path.basename(query_images[query_idx])
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

    # Create a figure for plotting
    fig, axs = plt.subplots(1, len(top_k_gallery_images) + 1, figsize=(15, 5))

    # Plot the query image
    axs[0].imshow(query_image)
    axs[0].set_title('Query Image')
    axs[0].axis('off')

    # Plot each of the top-k gallery images with their distance
    for i, (gallery_image_path, distance) in enumerate(zip(top_k_gallery_images, top_k_distances)):
        gallery_image = cv2.imread(gallery_image_path)
        gallery_image = cv2.cvtColor(gallery_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib
        axs[i + 1].imshow(gallery_image)
        axs[i + 1].set_title(f'Top {i + 1}\nDist: {distance:.2f}')
        axs[i + 1].axis('off')

    plt.tight_layout()

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the figure
    output_file_path = os.path.join(output_folder, f'{query_image_file_name}_top_{len(top_k_gallery_images)}.png')

    # Optionally show the images
    if plt_imgs:
        plt.show()
    else:
        plt.savefig(output_file_path)


    # Close the plot to free up memory
    plt.close(fig)

    




def extract_dataset_name(file_path):
    # Split the path by '/' and take the last part
    file_name = file_path.split('/')[-1]
    # Split the file name by underscores and take the relevant part
    parts = file_name.split('_')
    dataset_name = "_".join(parts[1:-2])  # Joining the parts that represent the dataset name
    return dataset_name


def calculate_average_ranks(top_k_dict):
    total_rank = 0.0
    num_query_images = len(top_k_dict)

    for query_image, gallery_images in top_k_dict.items():
        if query_image not in gallery_images:
            cur_rank = len(gallery_images) + 1
        else:
            for rank, gallery_image in enumerate(gallery_images, start=1):
                if gallery_image == query_image:
                    cur_rank = rank
                    break
                    
        total_rank += cur_rank

    return total_rank / num_query_images


def extract_reid_features(cfg, imgs_folder, model=None, model_path=None, num_classes=None):
    extractor = FeatureExtractor(
        cfg,
        # model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_classes=num_classes,
        model=model
    )

    current_path = os.getcwd()
    imgs_folder = os.path.join(current_path, imgs_folder)

    print('-'*5, '\n', imgs_folder)
    
    query_folder = os.path.join(imgs_folder, 'query')
    gallery_folder = os.path.join(imgs_folder, 'gallery')

    # Ensure the query and gallery folders exist
    if not os.path.exists(query_folder) or not os.path.exists(gallery_folder):
        raise FileNotFoundError("The specified query or gallery folder does not exist.")

    query_images = list_images_in_folder(query_folder)
    gallery_images = list_images_in_folder(gallery_folder)

    
    # Timing feature extraction for query images
    start_time_query = time.time()
    query_embeddings, query_visibility_scores = extract_part_based_features(
        extractor, query_images, batch_size=cfg.inference.batch_size)
    end_time_query = time.time()
    query_time = end_time_query - start_time_query

    # Timing feature extraction for gallery images
    start_time_gallery = time.time()
    gallery_embeddings, gallery_visibility_scores = extract_part_based_features(
        extractor, gallery_images, batch_size=cfg.inference.batch_size)
    end_time_gallery = time.time()
    gallery_time = end_time_gallery - start_time_gallery


    distmat, body_parts_distmat = compute_distance_matrix_using_bp_features(
        query_embeddings, gallery_embeddings, 
        query_visibility_scores, gallery_visibility_scores,
        dist_combine_strat = 'mean', use_gpu = False, metric = 'euclidean'
        )
        


    dataset_name = extract_dataset_name(cfg.model.load_weights)

    folder_name = os.path.join('outputs', dataset_name)
    clear_or_create_folder(folder_name)

    
    print(f"{len(query_images)} Query images embedding done with {len(query_images)/query_time:.2f} img per sec")
    print(f"{len(gallery_images)} Gallery images embedding done: {len(gallery_images)/gallery_time:.2f} img per sec")


    print(f'Total time embedding for + matching {time.time() - start_time_query:.2f} seconds' )


    top_k_dict = {}

    for query_idx, query_image_path in enumerate(query_images):
        # Get the top-k closest gallery images and their distances
        top_k_gallery_images, top_k_distances = get_top_k_closest_gallery_images(
            query_idx, distmat, gallery_images, k=cfg.inference.visrank_topk
        )

        # Save and visualize the query image and the top-k closest gallery images
        save_images(query_idx, query_images, top_k_gallery_images, top_k_distances, folder_name)

        # Extract only the file names (without the full path) for the dictionary
        query_image_name = os.path.basename(query_image_path)
        top_k_gallery_image_names = [os.path.basename(img) for img in top_k_gallery_images]

        # Store the mapping of query image to its top-k gallery images
        top_k_dict[query_image_name] = top_k_gallery_image_names



        # Assuming VID_PATH and dataset_name are defined elsewhere in your code
        vid_name_raw = os.path.basename(VID_PATH).split('.')[0]
        json_dir = 'outputs/jsons'
        json_file_path = os.path.join(json_dir, f'{vid_name_raw}.json')

        # Ensure the directory exists
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        # Calculate the average rank
        average_rank = calculate_average_ranks(top_k_dict)

        # Load or create the JSON file
        if os.path.exists(json_file_path):
            # Load existing JSON file
            with open(json_file_path, 'r') as json_file:
                json_dict = json.load(json_file)
        else:
            # Create a new JSON structure if the file doesn't exist
            json_dict = {}

        # Add or update the dataset_name with the average rank
        json_dict[dataset_name] = average_rank

        # Save the updated dictionary back to the JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(json_dict, json_file, indent=4)
