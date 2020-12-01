import argparse
import cv2
import os
from typing import List

from demo.record import AnnotationStore

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pore-id", type=int,
        help="Pore Id to retrieve"
    )
    parser.add_argument(
        "--image-folder",
        help="Path to directory containing original images or images with prediction overlays"
    )
    parser.add_argument(
        "--output",
        help="A directory to save retrieved pores to",
    )
    parser.add_argument(
        "--annotations",
        default="datasets/stoma/annotations/val.json",
        help="Path to file containing image ground truth annotations"
    )
    return parser.parse_args()

def load_image(filename:str):
    return cv2.imread(filename)

# Crop limit -> [x, y, w, h]
def crop_image(image, crop_limits: List[int]):
    x, y, w, h = crop_limits
    return  image[y : y + h, x : x + w]

def crop_pore_from_slide(image, bbox: List[int], offset:int=50):
    x = bbox[0] - offset if bbox[0] - offset > 0 else 0
    y = bbox[1] - offset if bbox[1] - offset > 0 else 0
    w = bbox[2] + 2*offset if bbox[2] + 2*offset < image.shape[0] else image.shape[0]
    h = bbox[3] + 2*offset if bbox[3] + 2*offset < image.shape[1] else image.shape[1]
    crop_limits = [x, y, w, h]
    return crop_image(image, crop_limits)

if __name__=='__main__':
    args = get_arguments()
    stoma_annotations = AnnotationStore(args.annotations, retrieval=True)
    # Recieve pore ID from user
    id_of_pore_to_retrieve = args.pore_id

    # Search pore ID in original image annotation .jsons to find its image
    file, gt = stoma_annotations.retrieve_pore(id_of_pore_to_retrieve)
    image_path = os.path.join(args.image_folder, file)
    # Crop pore from image with visualisation overlay (or not?)
    image = load_image(image_path)
    crop = crop_pore_from_slide(image, gt['bbox'])
    cv2.imwrite(str(id_of_pore_to_retrieve) + ".png", crop)
