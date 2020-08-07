import os
import json

import torch
import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from shapely import affinity
import shapely.geometry as shapes
import matplotlib.pyplot as plt
from rasterio.transform import IDENTITY
from mask_to_polygons.vectorification import geometries_from_mask

class AnnotationStore:
    def __init__(self, dir_annotations):
        self.ground_truth = self.load_ground_truth(dir_annotations)
        self.filename_id_map = self.create_filename_id_map(self.ground_truth)
        self.coco = COCO(dir_annotations)

    def load_ground_truth(self, gt_filepath):
        with open(gt_filepath) as file:
            raw_gt = json.load(file)
        # Associate image-id with filenames
        gt_formatted = dict()
        for image_details in raw_gt['images']:
            image_id = image_details['id']
            gt_formatted[image_id] = { 
                'filename' : image_details['file_name'],
                'annotations' : list() 
            }
        # Assign instance annotations to images
        for annotation in raw_gt['annotations']:
            image_id = annotation['image_id']
            gt_formatted[image_id]['annotations'].append(annotation)
        return gt_formatted

    def create_filename_id_map(self, ground_truth_dict):
        file_id_map = dict()
        for image_id in ground_truth_dict.keys():
            name = ground_truth_dict[image_id]['filename']
            file_id_map[name] = image_id
        return file_id_map

def record_predictions(predictions, filename, stoma_annotations):
    ground_truth = stoma_annotations.ground_truth
    filename_id_map = stoma_annotations.filename_id_map
    coco = stoma_annotations.coco

    image_name = filename.split('/')[-1]
    image_gt = ground_truth[filename_id_map[image_name]]['annotations']

    print(f"# of predictions: {len(predictions.pred_boxes)}")
    print(f"# of ground truth: {len(image_gt)}")
    remove_intersecting_predictions(predictions)
    image_json = { image_name : assign_preds_gt(predictions, image_gt, coco) }
    print(f"Matched: {len(image_json[image_name])}")

    with open('.'.join([filename[:-4], 'json']), 'w') as file:
        json.dump(image_json, file)

def remove_intersecting_predictions(predictions):
    for i, bbox_i in enumerate(predictions.pred_boxes):
        for j, bbox_j in enumerate(predictions.pred_boxes):
            if i == j: continue
            if intersects(bbox_i, bbox_j):
                bboxes = predictions.pred_boxes.tensor 
                if predictions.scores[i] >= predictions.scores[j]:
                    predictions.pred_boxes.tensor = torch.cat([bboxes[0:j], bboxes[j+1:]])
                else:
                    predictions.pred_boxes.tensor = torch.cat([bboxes[0:i], bboxes[i+1:]])

def assign_preds_gt(predictions, image_gt, coco):
    image_detections = []
    for i, bbox in enumerate(predictions.pred_boxes):
        for instance_gt in image_gt:
            if gt_intersects(bbox, instance_gt['bbox']):
                gt_pred_pair = create_pair(i, predictions, instance_gt, coco)
                image_detections.append(gt_pred_pair)
    return image_detections

def create_pair(i, predictions, instance_gt, coco):
    # Extract and format predictions
    pred_mask = predictions.pred_masks[i].cpu().numpy()
    pred_AB = predictions.pred_keypoints[i].flatten().tolist()

    if instance_gt['category_id'] == 1:
        # Processes prediction mask
        pred_polygon = mask_to_poly(pred_mask)
        pred_CD = find_CD(pred_polygon, pred_AB)
        pred_width = l2_dist(pred_CD)
        # Process ground truth mask
        gt_CD = find_CD(instance_gt['segmentation'][0], instance_gt['keypoints'])
        gt_width = l2_dist(gt_CD)
    else:
        # Placehold values for no prediction
        pred_polygon = []
        pred_CD = [-1,-1,1.-1,-1.1]
        pred_width = 0
        gt_CD = [-1,-1,1,-1,-1,1]
        gt_width = 0

    pred = { 
        'bbox' : predictions.pred_boxes[i].tensor.tolist()[0],
        'area' : pred_mask.sum().item(),
        'AB_keypoints' : pred_AB,
        'length' : l2_dist(pred_AB),
        'CD_keypoints' : pred_CD,
        'width' : pred_width,
        'class' : predictions.pred_classes[i].item(),
        'segmentation' : [pred_polygon]
    }
    
    # Add additional keys to ground truth
    instance_gt['width'] = gt_width
    instance_gt['CD_keypoinys'] = gt_CD
    instance_gt['area'] = calc_area(instance_gt, coco)

    detection_pair = {
        'gt' : instance_gt,
        'pred' : pred
    }
    return detection_pair

def find_CD(polygon, keypoints):
    # If no mask is predicted
    if len(polygon) < 1: return [-1,-1,1,-1,-1,1]
    # Convert to shapely linear ring
    x_points = [ polygon[i] for i in range(0, len(polygon), 2) ]
    y_points = [ polygon[i] for i in range(1, len(polygon), 2) ]
    polygon = [ [x, y] for x, y in zip(x_points, y_points) ]
    mask = shapes.LinearRing(polygon)
    # Find line perpendicular to AB
    A = shapes.Point(keypoints[0], keypoints[1])
    B = shapes.Point(keypoints[3], keypoints[4])

    l_AB = shapes.LineString([A, B])
    l_perp = affinity.rotate(l_AB, 90)
    #l_perp = affinity.scale(l_perp, 100, 100)
    # Find intersection with polygon
    intersections = l_perp.intersection(mask)

    # Visual check
    #plt.plot(*mask.xy)
    #plt.plot(*l_AB.xy)
    #plt.plot(*l_perp.xy)
    #plt.show()

    # If there is no intersection
    if intersections.is_empty:
        return [-1,-1,1,-1,-1,1]

    if intersections[0].coords.xy[1] > intersections[1].coords.xy[1]:
        D = intersections[0].coords.xy
        C = intersections[1].coords.xy
    else:
        D = intersections[1].coords.xy
        C = intersections[0].coords.xy
    return [*C[0], *C[1], 1, *D[0], *D[1], 1]

def l2_dist(keypoints):
    A, B = [keypoints[0], keypoints[1]], [keypoints[3], keypoints[4]]
    return pow((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2, 0.5)

def mask_to_poly(mask):
    if mask.sum() == 0: return []
    poly = geometries_from_mask(np.uint8(mask), IDENTITY, 'polygons')
    poly = poly[0]['coordinates'][0]
    flat_poly = []
    for point in poly:
        flat_poly.extend([point[0], point[1]])
    return flat_poly



# Convert polygon -> binary mask COCO.annToMask
#  Count pixels as area
def calc_area(gt_annotation, coco):
    gt_mask = torch.Tensor(coco.annToMask(gt_annotation)) == 1
    gt_area = gt_mask.sum()
    return gt_area.item()

def intersects(bbox_1, bbox_2):
    is_overlap = not (
        bbox_2[0] > bbox_1[2] or
        bbox_2[2] < bbox_1[0] or
        bbox_2[1] > bbox_1[3] or
        bbox_2[3] < bbox_1[1]
    )
    return is_overlap

def gt_intersects(bbox, gt_bbox):
    # GT boxes [x, y, width, height] -> [x1, y1, x2, y2]
    gt_bbox = [
        gt_bbox[0], gt_bbox[1],
        gt_bbox[0] + gt_bbox[2],
        gt_bbox[1] + gt_bbox[3]
    ]
    return intersects(bbox, gt_bbox)
