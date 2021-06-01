from scipy.stats import iqr
import numpy as np

from record import is_overlapping

CLOSE_TO_EDGE_DISTANCE = 20
CLOSE_TO_EDGE_SIZE_THRESHOLD = 0.85
SIZE_THRESHOLD = 0.3


def filter_invalid_predictions(predictions):
    remove_intersecting_predictions(predictions)
    remove_close_to_edge_detections(predictions)
    remove_extremley_small_detections(predictions)


def remove_intersecting_predictions(predictions):
    final_indices = []
    for i, bbox_i in enumerate(predictions.pred_boxes):
        intersecting = [
            j
            for j, bbox_j in enumerate(predictions.pred_boxes)
            if not i == j and is_overlapping(bbox_i, bbox_j)
        ]
        is_larger = [
            predictions.scores[i].item() >= predictions.scores[j].item()
            for j in intersecting
        ]
        if all(is_larger) or not intersecting:
            final_indices.append(i)
    
    select_predictions(predictions, final_indices)


def select_predictions(predictions, indices):
    predictions.pred_boxes.tensor = predictions.pred_boxes.tensor[indices]
    predictions.pred_classes = predictions.pred_classes[indices]
    predictions.pred_masks = predictions.pred_masks[indices]
    predictions.pred_keypoints = predictions.pred_keypoints[indices]


def remove_close_to_edge_detections(predictions):
    average_area = calculate_average_bbox_area(predictions)
    image_height,  image_width  = predictions.image_size
    final_indices = []
    for i, bbox_i in enumerate(predictions.pred_boxes):
        is_near_edge = is_bbox_near_edge(bbox_i, image_height, image_width)
        is_significantly_smaller_than_average = is_bbox_small(bbox_i, average_area)
        if is_near_edge and is_significantly_smaller_than_average:
            continue
        else:
            final_indices.append(i)
    select_predictions(predictions, final_indices)


def calculate_average_bbox_area(predictions):
    areas = [ calculate_bbox_area(bbox) for bbox in predictions.pred_boxes ]
    return sum(areas) / len(areas)


def calculate_bbox_area(bbox):
    width =  abs(bbox[2] - bbox[0])
    height = abs(bbox[3] - bbox[1])
    return width * height


def is_bbox_near_edge(bbox, image_height, image_width):
    x1, y1, x2, y2 = bbox
    is_near_edge = any([
        x1 < CLOSE_TO_EDGE_DISTANCE,
        y1 < CLOSE_TO_EDGE_DISTANCE,
        image_width - x2 < CLOSE_TO_EDGE_DISTANCE,
        image_height - y2 < CLOSE_TO_EDGE_DISTANCE,
    ])
    return is_near_edge

def is_bbox_small(bbox, average_area):
    threshold_area =  CLOSE_TO_EDGE_SIZE_THRESHOLD * average_area
    bbox_area = calculate_bbox_area(bbox)
    return bbox_area < threshold_area


def remove_extremley_small_detections(predictions):
    average_area = calculate_average_bbox_area(predictions)
    final_indices = []
    for i, bbox_i in enumerate(predictions.pred_boxes):
        if is_bbox_extremley_small(bbox_i, average_area):
            continue
        else:
            final_indices.append(i)
    select_predictions(predictions, final_indices)


def is_bbox_extremley_small(bbox, average_area):
    return calculate_bbox_area(bbox) < average_area * SIZE_THRESHOLD


def remove_image_outliers(predictions):
    predicted_lengths = get_lengths(predictions)
    inter_quartile_range = iqr(predicted_lengths, interpolation='midpoint')
    median = np.median(predicted_lengths)
    lower_whisker = median -  2.0 * inter_quartile_range
    higher_whisker = median +  2.0 * inter_quartile_range

    final_indices = []
    for i, length in enumerate(predicted_lengths):
        if lower_whisker <= length <= higher_whisker:
            final_indices.append(i)
    
    select_predictions(predictions, final_indices)
    predicted_lengths = get_lengths(predictions)

def get_lengths(predictions):
    lengths = []
    for keypoints in predictions.pred_keypoints:
        lengths.append(l2_dist(keypoints))
    return lengths

def l2_dist(keypoints):
    A, B = [keypoints[0][0], keypoints[0][1]], [keypoints[1][0], keypoints[1][1]]
    return pow((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2, 0.5)
