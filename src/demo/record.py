import json
from typing import Dict, List, Union

import numpy as np
import shapely
from shapely import affinity
import shapely.geometry as shapes
import matplotlib as mpl
import matplotlib.pyplot as plt
from rasterio.transform import IDENTITY
from mask_to_polygons.vectorification import geometries_from_mask
from detectron2.utils.visualizer import GenericMask

from datasets.create_annotations import NAMES_TO_CATEGORY_ID
from utils.bbox import is_bbox_a_in_bbox_b

Predicted_Lengths = []

"""
    Records predictions and ground truth pairs for an image as a JSON file
"""

IOU_THRESHOLD = 0.5
MINIMUM_LENGTH = 5


class AnnotationStore:
    def __init__(self, dir_annotations, retrieval=False):
        self.raw_gt = self.load_raw_gt(dir_annotations)
        self.ground_truth = self.init_ground_truth()
        self.filename_id_map = self.create_filename_id_map(self.ground_truth)
        self.ground_truth = self.populate_gt_with_annotations(self.ground_truth)
        if retrieval:
            self._pore_id_to_filename = self._create_pore_to_file_map(self.ground_truth)

    def load_raw_gt(self, gt_filepath):
        with open(gt_filepath) as file:
            raw_gt = json.load(file)
        return raw_gt

    def init_ground_truth(self):
        # Associate image-id with filenames
        gt_formatted = dict()
        for image_details in self.raw_gt["images"]:
            image_id = image_details["id"]
            gt_formatted[image_id] = {
                "filename": image_details["file_name"],
                "annotations": list(),
            }
        return gt_formatted

    def populate_gt_with_annotations(self, ground_truth):
        # Assign instance annotations to images
        for annotation in self.raw_gt["annotations"]:
            if "image_id" in annotation:
                image_id = annotation["image_id"]
                ground_truth[image_id]["annotations"].append(annotation)
            else:
                for label in annotation:
                    image_id = self.filename_id_map[label["image_name"] + ".png"]
                    ground_truth[image_id]["annotations"].append(label)
        return ground_truth

    def create_filename_id_map(self, ground_truth_dict):
        file_id_map = dict()
        for image_id in ground_truth_dict.keys():
            name = ground_truth_dict[image_id]["filename"]
            file_id_map[name] = image_id
        return file_id_map

    def _create_pore_to_file_map(self, gt_dict):
        pore_to_file_map = dict()
        for image_id in gt_dict:
            image_gt = gt_dict[image_id]
            file = image_gt["filename"]
            for pore_gt in image_gt["annotations"]:
                pore_to_file_map[pore_gt["id"]] = file
        return pore_to_file_map

    def retrieve_pore(self, pore_id):
        filename = self._pore_id_to_filename[pore_id]
        image_id = self.filename_id_map[filename]
        image_annotations = self.ground_truth[image_id]["annotations"]

        for image_annotation in image_annotations:
            if image_annotation["id"] == pore_id:
                pore_annotation = image_annotation
                break
        return [filename, pore_annotation]


def np_encoder(object):
    if isinstance(object, (np.generic, np.ndarray)):
        return object.item()


def record_predictions(predictions, filename, stoma_annotations):
    if stoma_annotations is None:
        print(f"# of predictions: {len(predictions.pred_boxes)}")
        predictions = convert_predictions_to_list_of_dictionaries(predictions)
        with open(".".join([filename[:-4] + "-predictions", "json"]), "w") as file:
            json.dump({"detections": predictions}, file, default=np_encoder)
        return predictions

    predictions = convert_predictions_to_list_of_dictionaries(predictions)

    ground_truth = stoma_annotations.ground_truth
    filename_id_map = stoma_annotations.filename_id_map

    image_name = filename.split("/")[-1]
    image_gt = ground_truth[filename_id_map[image_name]]["annotations"]

    print(f"# of ground truth: {len(image_gt)}")
    image_gt = convert_gt_to_list_of_dictionaries(image_gt)
    pairs = assign_preds_gt(predictions, image_gt)
    image_json = {"detections": pairs}
    print(f"Matched: {len(image_json['detections'])}")

    with open(".".join([filename[:-4], "json"]), "w") as file:
        json.dump(image_json, file)
    with open(".".join([filename[:-4] + "-gt", "json"]), "w") as file:
        json.dump({"detections": image_gt}, file)


def assign_preds_gt(predictions, image_gt):
    gt_prediction_pairs = []
    for i, prediction in enumerate(predictions):
        bbox = prediction["bbox"]
        gt_prediction_pairs.extend(
            [
                create_pair(predictions[i], instance_gt)
                for instance_gt in image_gt
                if gt_overlaps(bbox, instance_gt["bbox"])
            ]
        )
    return gt_prediction_pairs


def convert_gt_to_list_of_dictionaries(image_gt):
    image_gt = [convert_gt_to_dictionary(instance_gt) for instance_gt in image_gt]
    return image_gt


def convert_predictions_to_list_of_dictionaries(predictions):
    predictions = [
        convert_predictions_to_dictionary(i, predictions)
        for i in range(len(predictions.pred_boxes))
        if is_stomata_complex(i, predictions)
    ]
    return predictions


def is_stomata_complex(i, predictions) -> bool:
    class_label = get_predicted_class(i, predictions)
    complex_categories = [
        NAMES_TO_CATEGORY_ID["Closed Stomata"],
        NAMES_TO_CATEGORY_ID["Open Stomata"],
    ]
    return class_label in complex_categories


def get_predicted_class(i, predictions) -> int:
    return predictions.pred_classes[i].item()


def get_predicted_mask(i, predictions) -> np.array:
    mask = predictions.pred_masks[i].cpu().numpy()
    return GenericMask(mask, *mask.shape)


def get_predicted_keypoints(i, predictions) -> List[int]:
    return predictions.pred_keypoints[i].flatten().tolist()


def get_predicted_bounding_box(i, predictions) -> List[int]:
    return predictions.pred_boxes[i].tensor.tolist()[0]


def get_prediction_confidence(i, predictions) -> float:
    return predictions.scores[i].item()


def convert_predictions_to_dictionary(i, predictions):
    # Extract and format predictions
    class_label = get_predicted_class(i, predictions)
    guard_cell_mask = get_predicted_mask(i, predictions)
    keypoints_AB = get_predicted_keypoints(i, predictions)
    pore_length = l2_dist(keypoints_AB)

    guard_cell_area = guard_cell_mask.area()
    if len(guard_cell_mask.polygons) > 1:
        guard_cell_polygon = {
            "external": guard_cell_mask.polygons[0].tolist(),
            "internal": guard_cell_mask.polygons[1].tolist(),
        }
    else:
        guard_cell_polygon = {
            "external": guard_cell_mask.polygons[0].tolist(),
            "internal": [],
        }

    if class_label == NAMES_TO_CATEGORY_ID["Open Stomata"]:
        pore_dict = get_stomatal_pore(i, predictions)
        if pore_dict is None:
            class_label = NAMES_TO_CATEGORY_ID["Closed Stomata"]
        else:
            pore_polygon = pore_dict["mask"].polygons[0].tolist()
            # Processes prediction mask
            keypoints_CD = find_CD(pore_polygon, keypoints_AB, gt=False)
            pore_width = l2_dist(keypoints_CD)
            pore_area = pore_dict["mask"].area()
            if pore_length < MINIMUM_LENGTH:
                x_points = [x for x in pore_polygon[0::2]]
                y_points = [y for y in pore_polygon[1::2]]
                keypoints_AB = extract_polygon_AB(x_points, y_points)
                pore_length = l2_dist(keypoints_AB)

    if class_label == NAMES_TO_CATEGORY_ID["Closed Stomata"]:
        keypoints_CD = [-1, -1, 1.0 - 1, -1, 1]
        pore_width = 0
        pore_area = 0
        pore_polygon = []

    subsidiary_polygons = []
    subsidiary_cells = get_subsidiary_cells(i, predictions)
    for subsidiary_cell in subsidiary_cells:
        polygon = subsidiary_cell["mask"].polygons[0].tolist()
        subsidiary_polygons.append(polygon)

    prediction_dict = {
        "bbox": get_predicted_bounding_box(i, predictions),
        "pore_area": pore_area,
        "pore_length": pore_length,
        "pore_width": pore_width,
        "pore_polygon": pore_polygon,
        "AB_keypoints": keypoints_AB,
        "CD_keypoints": keypoints_CD,
        "category_id": class_label,
        "guard_cell_area": guard_cell_area,
        "guard_cell_polygon": guard_cell_polygon,
        "subsidiary_cell_polygons": subsidiary_polygons,
        "confidence": get_prediction_confidence(i, predictions),
    }
    Predicted_Lengths.append(pore_length)
    return prediction_dict


def get_stomatal_pore(i, predictions) -> Dict:
    pore_index = find_stomatal_pore(i, predictions)
    if pore_index is not None:
        return format_pore_prediction(pore_index, predictions)
    return None


def find_stomatal_pore(i, predictions) -> Union[int, None]:
    stomata_bbox = get_predicted_bounding_box(i, predictions)
    for j in range(len(predictions.pred_boxes)):
        if is_stomatal_pore(j, predictions):
            pore_bbox = get_predicted_bounding_box(j, predictions)
            if is_bbox_a_in_bbox_b(pore_bbox, stomata_bbox):
                return j
    return None


def is_stomatal_pore(i, predictions) -> bool:
    class_label = get_predicted_class(i, predictions)
    return class_label == NAMES_TO_CATEGORY_ID["Stomatal Pore"]


def format_pore_prediction(i, predictions) -> Dict:
    pore_dict = {
        "bbox": get_predicted_bounding_box(i, predictions),
        "mask": get_predicted_mask(i, predictions),
    }
    return pore_dict


def get_subsidiary_cells(i, predictions) -> List[Dict]:
    subsidiary_cells = []
    subsidiary_cell_indices = find_subsidiary_cells(i, predictions)
    if len(subsidiary_cell_indices) > 0:
        for index in subsidiary_cell_indices:
            subsidiary_cells.append(format_subsidiary_cell(index, predictions))
    return subsidiary_cells


def find_subsidiary_cells(i, predictions) -> Union[int, None]:
    subsidiary_cell_indices = []
    stomata_bbox = get_predicted_bounding_box(i, predictions)
    for j in range(len(predictions.pred_boxes)):
        if is_subsidiary_cell(j, predictions):
            cell_bbox = get_predicted_bounding_box(j, predictions)
            if is_bbox_a_in_bbox_b(cell_bbox, stomata_bbox):
                subsidiary_cell_indices.append(j)
    return subsidiary_cell_indices


def is_subsidiary_cell(i, predictions) -> bool:
    class_label = get_predicted_class(i, predictions)
    return class_label == NAMES_TO_CATEGORY_ID["Subsidiary cells"]


def format_subsidiary_cell(i, predictions) -> Dict:
    subsidiary_cell_dict = {
        "bbox": get_predicted_bounding_box(i, predictions),
        "mask": get_predicted_mask(i, predictions),
    }
    return subsidiary_cell_dict


def convert_gt_to_dictionary(instance_gt):
    if instance_gt["category_id"] == 1:
        # Process ground truth mask
        gt_CD = find_CD(instance_gt["segmentation"][0], instance_gt["keypoints"])
        gt_width = l2_dist(gt_CD)
        instance_gt["area"] = calc_area(instance_gt)
    else:
        # Placehold values for no prediction
        gt_width = 0
        gt_CD = [-1, -1, 1, -1, -1, 1]
        instance_gt["area"] = 0
    # Add additional keys to ground truth
    instance_gt["width"] = gt_width
    instance_gt["CD_keypoints"] = gt_CD
    instance_gt["length"] = l2_dist(instance_gt["keypoints"])
    return instance_gt


def create_pair(predictions, instance_gt):
    detection_pair = {"gt": instance_gt, "pred": predictions}
    return detection_pair


def find_CD(polygon, keypoints=None, gt=True):
    # If no mask is predicted
    if len(polygon) < 1:
        #    counter += 1
        return [-1, -1, 1, -1, -1, 1]

    x_points = [x for x in polygon[0::2]]
    y_points = [y for y in polygon[1::2]]

    if keypoints == None:
        keypoints = extract_polygon_AB(x_points, y_points)
    # Convert to shapely linear ring
    polygon = [[x, y] for x, y in zip(x_points, y_points)]
    mask = shapes.LinearRing(polygon)
    # Find line perpendicular to AB
    A = shapes.Point(keypoints[0], keypoints[1])
    B = shapes.Point(keypoints[3], keypoints[4])

    l_AB = shapes.LineString([A, B])
    l_perp = affinity.rotate(l_AB, 90)
    l_perp = affinity.scale(l_perp, 10, 10)
    # Find intersection with polygon
    try:
        intersections = l_perp.intersection(mask)
    except:
        intersections = shapes.collection.GeometryCollection()

    """
    # Visual check
    plt.plot(*mask.xy)
    plt.plot(*l_AB.xy)
    plt.plot(*l_perp.xy)
    if counter % 2 == 0:
        plt.savefig('visualised_plots/out-'+str(int(counter/2))+'.png')
        plt.clf()
    elif gt:
        plt.clf()
    counter += 1
    """
    # Invalid intersection
    if intersections.is_empty or len(intersections) == 1:
        return [-1, -1, 1, -1, -1, 1]

    if intersections[0].coords.xy[1] > intersections[1].coords.xy[1]:
        D = intersections[0].coords.xy
        C = intersections[1].coords.xy
    else:
        D = intersections[1].coords.xy
        C = intersections[0].coords.xy
    return [C[0][0], C[1][0], 1, D[0][0], D[1][0], 1]


def l2_dist(keypoints):
    A, B = [keypoints[0], keypoints[1]], [keypoints[3], keypoints[4]]
    return pow((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2, 0.5)


def mask_to_poly(mask):
    if mask.sum() == 0:
        return []
    poly = geometries_from_mask(np.uint8(mask), IDENTITY, "polygons")
    if len(poly) > 1:
        poly = find_maximum_area_polygon(poly)
    else:
        poly = poly[0]
    poly = poly["coordinates"][0]
    return flatten_polygon(poly)


def find_maximum_area_polygon(polygons):
    maximum = 0
    for i, polygon in enumerate(polygons):
        try:
            polygon = shapely.geometry.Polygon(polygon["coordinates"][0])
        except:
            continue
        if polygon.area > maximum:
            maximum = polygon.area
            index = i
    return polygons[index]


def flatten_polygon(polygon):
    flat_polygon = []
    for point in polygon:
        flat_polygon.extend([point[0], point[1]])
    return flat_polygon


def calc_area(gt_annotation):
    polygon = gt_annotation["segmentation"][0]
    x_points = [x for x in polygon[0::2]]
    y_points = [y for y in polygon[1::2]]
    polygon = [[x, y] for x, y in zip(x_points, y_points)]
    polygon = shapely.geometry.Polygon(polygon)
    gt_area = polygon.area
    return gt_area


def intersects(bbox_1, bbox_2) -> bool:
    is_overlap = not (
        bbox_2[0] > bbox_1[2]
        or bbox_2[2] < bbox_1[0]
        or bbox_2[1] > bbox_1[3]
        or bbox_2[3] < bbox_1[1]
    )
    return is_overlap


def is_overlapping(bbox1, bbox2) -> bool:
    if intersects(bbox1, bbox2):
        return is_overlap(bbox1, bbox2, IOU_THRESHOLD)
    return False


def is_overlap(bbox_1, bbox_2, threshold) -> bool:
    iou = intersection_over_union(bbox_1, bbox_2)
    if iou > threshold:
        return True
    return False


def gt_overlaps(bbox, gt_bbox) -> bool:
    # GT boxes [x, y, width, height] -> [x1, y1, x2, y2]
    gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]

    return is_overlapping(bbox, gt_bbox)


def intersection_over_union(bbox, gt_bbox):
    intersecting_area = calculate_area_of_intersection(bbox, gt_bbox)
    pred_area = calculate_area_of_bbox(bbox)
    gt_area = calculate_area_of_bbox(gt_bbox)
    iou = intersecting_area / float(pred_area + gt_area - intersecting_area)
    return iou


def calculate_area_of_intersection(bbox, gt_bbox) -> float:
    x_max, y_max = max(bbox[0], gt_bbox[0]), max(bbox[1], gt_bbox[1])
    x_min, y_min = min(bbox[2], gt_bbox[2]), min(bbox[3], gt_bbox[3])
    return max(0, x_min - x_max + 1) * max(0, y_min - y_max + 1)


def calculate_area_of_bbox(bbox) -> float:
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)


def extract_polygon_AB(x_values, y_values):
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_extent = x_max - x_min
    y_extent = y_max - y_min
    # Enables pores of arbitrary orientation
    if x_extent > y_extent:
        major_axis_values = x_values
        minor_axis_values = y_values
        maximum_major_value = x_max
        minimum_major_value = x_min
    else:
        major_axis_values = y_values
        minor_axis_values = x_values
        maximum_major_value = y_max
        minimum_major_value = y_min
    # Left/Right along major axis
    left_hand_values, right_hand_values = [], []
    for i, minor_value in enumerate(minor_axis_values):
        if maximum_major_value == major_axis_values[i]:
            right_hand_values.append(minor_value)
        if minimum_major_value == major_axis_values[i]:
            left_hand_values.append(minor_value)
    # Use midpoint of extreme values as keypoint value
    right_hand_value = (right_hand_values[0] + right_hand_values[-1]) / 2
    left_hand_value = (left_hand_values[0] + left_hand_values[-1]) / 2

    if x_extent > y_extent:
        keypoints = [
            minimum_major_value,
            left_hand_value,
            1,
            maximum_major_value,
            right_hand_value,
            1,
        ]
    else:
        keypoints = [
            left_hand_value,
            minimum_major_value,
            1,
            right_hand_value,
            maximum_major_value,
            1,
        ]
    return keypoints


def draw_width_predictions(detectron_vis, predictions):
    image = detectron_vis.get_image()
    fig, ax = setup_plot(image)
    if predictions is not None:
        keypoints(ax, predictions)
    return fig


def setup_plot(image):
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(image)
    return fig, ax


def keypoints(mpl_axis, annotations):
    for annotation in annotations:
        draw_width_keypoints(mpl_axis, annotation)


def draw_width_keypoints(mpl_axis, annotation):
    keypoints_width = annotation["CD_keypoints"]
    # Dummy value for closed stoma -> no width
    if keypoints_width[0] == -1:
        return
    keypoints_x = extract_x_keypoints(keypoints_width)
    keypoints_y = extract_y_keypoints(keypoints_width)
    draw_points_and_lines(mpl_axis, keypoints_x, keypoints_y)


def extract_x_keypoints(keypoints):
    return [keypoints[0], keypoints[3]]


def extract_y_keypoints(keypoints):
    return [keypoints[1], keypoints[4]]


def draw_points_and_lines(mpl_axis, keypoints_x, keypoints_y):
    colour = "red"
    draw_lines(mpl_axis, [keypoints_x, keypoints_y], colour)
    draw_points(mpl_axis, zip(keypoints_x, keypoints_y), colour)


def draw_points(mpl_axis, keypoints, colour, radius=2):
    for keypoint in keypoints:
        mpl_axis.add_patch(
            mpl.patches.Circle(
                keypoint,
                radius=radius,
                fill=True,
                color=colour,
            )
        )


def draw_lines(mpl_axis, keypoints, colour):
    mpl_axis.add_line(
        mpl.lines.Line2D(
            keypoints[0],
            keypoints[1],
            linewidth=0.5,
            color=colour,
        )
    )
