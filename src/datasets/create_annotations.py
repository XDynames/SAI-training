import copy
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union


import xmltodict
from loguru import logger
from PIL import Image
from tqdm import tqdm


from datasets.arguments import get_parser
from utils.bbox import is_bbox_a_in_bbox_b

BOUNDING_BOX_PADDING = 10
NAMES_TO_CATEGORY_ID = {
    "Closed Stomata": 0,
    "Open Stomata": 1,
    "Stomatal Pore": 2,
    "Subsidiary cells": 3,
}


class AnnotationCoverter:
    def __init__(self, config: Namespace):
        self._setup(config)

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def n_train_samples(self) -> int:
        return len(self._training_samples)

    @property
    def n_validation_samples(self) -> int:
        return len(self._validation_samples)

    def convert_annotations(self):
        self._reset()
        self._convert_training_samples()
        self._save_coco_annotations("train.json")
        self._reset()
        self._convert_validation_samples()
        self._save_coco_annotations("val.json")

    def _reset(self):
        self._image_details, self._annotations = [], []
        self._annotation_id, self._image_id = 0, 0
        self._reset_annotations()

    def _reset_annotations(self):
        self._annotations_by_type = defaultdict(list)
        self._stomata_bboxes_xyxy = []

    def _convert_training_samples(self):
        self._convert_samples(self._training_samples)
        logger.info(f"Created Training Set with {len(self._annotations)} instances")

    def _convert_validation_samples(self):
        self._convert_samples(self._validation_samples)
        logger.info(f"Created Validation Set with {len(self._annotations)} instances")

    def _convert_samples(self, samples: List[Dict]):
        for sample in tqdm(samples):
            self._add_image_details_to_split(sample)
            self._add_annotation_details_to_split(sample)
            self._image_id += 1
            self._reset_annotations()

    def _save_coco_annotations(self, filename: str):
        to_save = {
            "images": self._image_details,
            "annotations": self._annotations,
            "categories": [
                {"id": NAMES_TO_CATEGORY_ID[name], "name": name}
                for name in NAMES_TO_CATEGORY_ID.keys()
            ],
        }
        output_path = self._output_path_root.joinpath(filename)
        with output_path.open("w") as file:
            json.dump(to_save, file)
        logger.info(f"Saved annotations to: {output_path}")

    def _add_image_details_to_split(self, sample: Dict):
        image_name = self._get_image_name(sample["image"])
        width, height = self._get_image_dimensions(sample["image"])
        image_info = {
            "id": self._image_id,
            "file_name": image_name,
            "height": height,
            "width": width,
        }
        self._image_details.append(image_info)

    def _get_image_dimensions(self, image_path: Path) -> List[int]:
        with Image.open(image_path) as image:
            width, height = image.size
        return [width, height]

    def _get_image_name(self, image_path: Path) -> str:
        return image_path.stem + image_path.suffix

    def _add_annotation_details_to_split(self, sample: Dict):
        annotations = self._load_image_annotations(sample["annotations"])
        self._collect_annotations_by_types(annotations)
        self._init_stomata_bounding_boxes()
        self._create_annotations()

    def _load_image_annotations(self, annotation_path: Path) -> List[Dict]:
        with annotation_path.open() as file:
            image_annotations = xmltodict.parse(file.read())
        return image_annotations["annotation"]["object"]

    def _collect_annotations_by_types(self, annotations: List[Dict]):
        for annotation in annotations:
            annotation_name = annotation["name"]
            if self._is_invalid_annotation(annotation):
                continue
            self._annotations_by_type[annotation_name].append(annotation)

    def _is_invalid_annotation(self, annotation: Dict) -> bool:
        is_difficult = annotation["difficult"] == "1"
        is_difficult = is_difficult and self._is_filtering_difficult_samples
        is_truncated = annotation["truncated"] == "1"
        is_truncated = is_truncated and self._is_filtering_truncated_samples
        return is_truncated or is_difficult

    def _init_stomata_bounding_boxes(self):
        stomata_annotations = copy.copy(self._annotations_by_type["Open Stomata"])
        stomata_annotations.extend(self._annotations_by_type["Closed Stomata"])
        for annotation in stomata_annotations:
            bounding_box = self._bbox_dict_to_xyxy(annotation["bndbox"])
            self._stomata_bboxes_xyxy.append(bounding_box)

    def _bbox_dict_to_xyxy(self, to_convert: Dict) -> List[float]:
        bounding_box = [
            float(to_convert["xmin"]),
            float(to_convert["ymin"]),
            float(to_convert["xmax"]),
            float(to_convert["ymax"]),
        ]
        return bounding_box

    def _create_annotations(self):
        self._create_pore_annotations()
        # self._create_guard_cell_annotations()
        self._create_open_stomata_annotations()
        self._create_closed_stomata_annotations()
        self._create_subsidiary_annotations()

    def _create_pore_annotations(self):
        # Adds each open pore as an individual instance
        self._add_annotations_to_dataset("Stomatal Pore")

    def _create_guard_cell_annotations(self):
        # Pairs the cells of this type into a single instance
        # self._assign_bounding_box_to_annotation("Guard cells")
        # Adds each cell in the pair as individual instances
        self._add_annotations_to_dataset("Guard cells")

    def _create_subsidiary_annotations(self):
        # Pairs the cells of this type into a single instance
        # self._assign_bounding_box_to_annotation("Subsidiary cells")
        # Adds each cell in the pair as individual instances
        self._add_annotations_to_dataset("Subsidiary cells")

    def _create_closed_stomata_annotations(self):
        for annotation in self._annotations_by_type["Closed Stomata"]:
            guard_cell_annotations = self._maybe_find_guard_cells_of_stomata(annotation)
            pore_annotation = self._maybe_find_pore_of_stomata(annotation)
            if guard_cell_annotations is not None and pore_annotation is not None:
                self._add_closed_stomata_to_dataset(
                    annotation,
                    guard_cell_annotations,
                    pore_annotation,
                )

    def _maybe_find_pore_of_stomata(self, annotation):
        pore_annotation = self._find_pore_of_stomata(annotation)
        if pore_annotation is None:
            logger.info(f"Didn't Find a pore for stomata {annotation}")
        return pore_annotation

    def _maybe_find_guard_cells_of_stomata(self, annotation):
        guard_cell_annotation = self._find_guard_cells_of_stomata(annotation)
        if len(guard_cell_annotation) != 2:
            logger.info(f"Didn't Find a guard cell annotation for stomata {annotation}")
            return None
        return guard_cell_annotation

    def _find_guard_cells_of_stomata(self, annotation: Dict) -> Dict:
        guard_cell_annotations = []
        stomata_bbox = self._bbox_dict_to_xyxy(annotation["bndbox"])
        stomata_bbox = self._add_padding_to_bounding_box(stomata_bbox)

        for guard_cell_annotation in self._annotations_by_type["Guard cells"]:
            guard_cell_bbox = self._bbox_dict_to_xyxy(guard_cell_annotation["bndbox"])
            if is_bbox_a_in_bbox_b(guard_cell_bbox, stomata_bbox):
                guard_cell_annotations.append(guard_cell_annotation)
        return guard_cell_annotations

    def _add_closed_stomata_to_dataset(
        self,
        stomata: Dict,
        guard_cells: List[Dict],
        pore: Dict,
    ):
        xyxy_bbox = self._bbox_dict_to_xyxy(stomata["bndbox"])
        bounding_box = self._xyxy_to_xywh(xyxy_bbox)
        keypoints = self._get_closed_stomata_keypoints(pore["line"])
        segmentation = [
            self._annotation_to_polygon(guard_cell) for guard_cell in guard_cells
        ]
        converted_annotation = {
            "bbox": bounding_box,
            "area": self._get_bbox_area(bounding_box),
            "iscrowd": 0,
            "category_id": NAMES_TO_CATEGORY_ID["Closed Stomata"],
            "segmentation": segmentation,
            "num_keypoints": 2,
            "keypoints": keypoints,
            "image_id": self._image_id,
            "id": self._annotation_id,
        }
        self._annotations.append(converted_annotation)
        self._annotation_id += 1

    def _line_to_polygon(self, line: List[float]) -> List[float]:
        polygon = [
            line[0],
            line[1],
            line[0] + 1,
            line[1] + 1,
            line[2] + 1,
            line[3] + 1,
            line[2],
            line[3],
        ]
        return polygon

    def _get_closed_stomata_keypoints(self, polygon: List[str]) -> List[float]:
        polygon = [float(item) for item in polygon.values()]
        if polygon[0] < polygon[2]:
            keypoints = [polygon[0], polygon[1], 1, polygon[2], polygon[3], 1]
        else:
            keypoints = [polygon[2], polygon[3], 1, polygon[0], polygon[1], 1]
        return keypoints

    def _create_open_stomata_annotations(self):
        for annotation in self._annotations_by_type["Open Stomata"]:
            guard_cell_annotations = self._maybe_find_guard_cells_of_stomata(annotation)
            pore_annotation = self._maybe_find_pore_of_stomata(annotation)
            if guard_cell_annotations is not None and pore_annotation is not None:
                self._add_open_stomata_to_dataset(
                    annotation,
                    guard_cell_annotations,
                    pore_annotation,
                )

    def _add_open_stomata_to_dataset(
        self,
        stomata: Dict,
        guard_cells: List[Dict],
        pore: Dict,
    ):
        xyxy_bbox = self._bbox_dict_to_xyxy(stomata["bndbox"])
        bounding_box = self._xyxy_to_xywh(xyxy_bbox)
        keypoints = self._get_open_stomata_keypoints(pore["polygon"])
        segmentation = [
            self._annotation_to_polygon(guard_cell) for guard_cell in guard_cells
        ]
        converted_annotation = {
            "bbox": bounding_box,
            "area": self._get_bbox_area(bounding_box),
            "iscrowd": 0,
            "category_id": NAMES_TO_CATEGORY_ID["Open Stomata"],
            "segmentation": segmentation,
            "num_keypoints": 2,
            "keypoints": keypoints,
            "image_id": self._image_id,
            "id": self._annotation_id,
        }
        self._annotations.append(converted_annotation)
        self._annotation_id += 1

    def _annotation_to_polygon(self, annotation) -> List[float]:
        return [float(item) for item in annotation["polygon"].values()]

    def _get_open_stomata_keypoints(self, polygon: List[str]) -> List[float]:
        polygon = [float(item) for item in polygon.values()]
        xs = polygon[::2]
        ys = polygon[1::2]
        xmin = min(range(len(xs)), key=xs.__getitem__)
        xmax = max(range(len(xs)), key=xs.__getitem__)
        ymin = min(range(len(ys)), key=ys.__getitem__)
        ymax = max(range(len(ys)), key=ys.__getitem__)
        y_extent = ys[ymax] - ys[ymin]
        x_extent = xs[xmax] - xs[xmin]
        if x_extent > y_extent:
            keypoints = [
                polygon[xmin * 2],
                polygon[xmin * 2 + 1],
                1,
                polygon[xmax * 2],
                polygon[xmax * 2 + 1],
                1,
            ]
        else:
            keypoints = [
                polygon[ymin * 2],
                polygon[ymin * 2 + 1],
                1,
                polygon[ymax * 2],
                polygon[ymax * 2 + 1],
                1,
            ]
        return keypoints

    def _find_pore_of_stomata(self, annotation: Dict) -> Dict:
        stomata_bbox = self._bbox_dict_to_xyxy(annotation["bndbox"])
        for pore_annotation in self._annotations_by_type["Stomatal Pore"]:
            pore_bbox = self._bbox_dict_to_xyxy(pore_annotation["bndbox"])
            if is_bbox_a_in_bbox_b(pore_bbox, stomata_bbox):
                return pore_annotation
        return None

    def _add_annotations_to_dataset(self, key: str):
        for annotation in self._annotations_by_type[key]:
            if "line" in annotation:
                continue
            self._add_annotation_to_dataset(annotation, key)

    def _assign_bounding_box_to_annotation(self, key: str):
        annotations = defaultdict(list)
        for annotation in self._annotations_by_type[key]:
            # Find paris of guard and subsidiary cells and add them to the same annotation
            index = self._find_stomata_bbox(annotation)
            if index is None:
                logger.info(f"Didn't Match {annotation}")
                continue
            annotations[index].append(annotation)
        self._add_paired_annotations_to_dataset(annotations, key)

    def _find_stomata_bbox(
        self,
        annotation: Dict,
    ) -> Union[List[float], None]:
        for i, bounding_box in enumerate(self._stomata_bboxes_xyxy):
            annotation_bbox = self._bbox_dict_to_xyxy(annotation["bndbox"])
            if is_bbox_a_in_bbox_b(annotation_bbox, bounding_box):
                return i
        return None

    def _add_paired_annotations_to_dataset(self, annotations: Dict, key: str):
        for bbox_index, pair in annotations.items():
            self._add_paired_annotation_to_dataset(bbox_index, pair, key)

    def _add_paired_annotation_to_dataset(
        self,
        bbox_index: int,
        pair: List[Dict],
        key: str,
    ):
        bounding_box = self._calculate_bounding_box_of_pair(pair)
        segmentation = []
        for annotation in pair:
            mask = [float(value) for value in annotation["polygon"].values()]
            segmentation.append(mask)
        converted_annotation = {
            "bbox": bounding_box,
            "area": self._get_bbox_area(bounding_box),
            "iscrowd": 0,
            "category_id": NAMES_TO_CATEGORY_ID[key],
            "segmentation": segmentation,
            "num_keypoints": 2,
            "keypoints": [0, 0, 0, 0, 0, 0],
            "image_id": self._image_id,
            "id": self._annotation_id,
        }
        self._annotations.append(converted_annotation)
        self._annotation_id += 1

    def _calculate_bounding_box_of_pair(self, pair: List[Dict]) -> List[float]:
        if len(pair) == 1:
            bounding_box = self._bbox_dict_to_xyxy(pair[0]["bndbox"])
            return self._xyxy_to_xywh(bounding_box)
        bounding_box_1 = self._bbox_dict_to_xyxy(pair[0]["bndbox"])
        bounding_box_2 = self._bbox_dict_to_xyxy(pair[1]["bndbox"])
        bounding_box = self._calculate_bbox_of_bboxes(bounding_box_1, bounding_box_2)
        return self._xyxy_to_xywh(bounding_box)

    def _calculate_bbox_of_bboxes(
        self,
        bbox_1: List[float],
        bbox_2: List[float],
    ) -> List[float]:
        bounding_box = [
            min(bbox_1[0], bbox_2[0]),
            min(bbox_1[1], bbox_2[1]),
            max(bbox_1[2], bbox_2[2]),
            max(bbox_1[3], bbox_2[3]),
        ]
        return bounding_box

    def _add_annotation_to_dataset(
        self,
        annotation: Dict,
        key: str,
    ):
        bounding_box = self._bbox_dict_to_xyxy(annotation["bndbox"])
        bounding_box = self._xyxy_to_xywh(bounding_box)
        bounding_box = self._add_padding_to_bounding_box(bounding_box)
        converted_annotation = {
            "bbox": bounding_box,
            "area": self._get_bbox_area(bounding_box),
            "iscrowd": 0,
            "category_id": NAMES_TO_CATEGORY_ID[key],
            "segmentation": [self._annotation_to_polygon(annotation)],
            "num_keypoints": 2,
            "keypoints": [0, 0, 0, 0, 0, 0],
            "image_id": self._image_id,
            "id": self._annotation_id,
        }
        self._annotations.append(converted_annotation)
        self._annotation_id += 1

    def _xyxy_to_xywh(self, xyxy_bbox: List[float]) -> List[float]:
        xywh_bbox = [
            xyxy_bbox[0],
            xyxy_bbox[1],
            xyxy_bbox[2] - xyxy_bbox[0],
            xyxy_bbox[3] - xyxy_bbox[1],
        ]
        return xywh_bbox

    def _add_padding_to_bounding_box(self, bounding_box: List[float]) -> List[float]:
        bounding_box = [
            bounding_box[0] - BOUNDING_BOX_PADDING,
            bounding_box[1] - BOUNDING_BOX_PADDING,
            bounding_box[2] + BOUNDING_BOX_PADDING,
            bounding_box[3] + BOUNDING_BOX_PADDING,
        ]
        return bounding_box

    def _get_bbox_area(self, xyhw_bbox: List[float]) -> float:
        return xyhw_bbox[2] * xyhw_bbox[3]

    def _setup(self, config: Namespace):
        self._init_config_attributes(config)
        self._maybe_create_output_folder()
        self._create_sample_lists()
        self._log_successful_setup()

    def _init_config_attributes(self, config: Namespace):
        self._original_annotation_path = Path(config.annotation_path)
        self._original_image_path = Path(config.image_path)
        self._output_path_root = Path(config.output_path)
        self._n_train_samples = config.num_train
        self._is_shuffled_splits = config.shuffled_splits
        self._is_filtering_difficult_samples = True
        self._is_filtering_truncated_samples = True

    def _maybe_create_output_folder(self):
        self._output_path_root.mkdir(parents=True, exist_ok=True)

    def _create_sample_lists(self):
        self._init_sample_list()
        self._generate_split_indices()
        self._select_samples()

    def _init_sample_list(self):
        images = sorted(self._get_all_images())
        annotations = sorted(self._original_annotation_path.glob("*.xml"))
        samples = [
            {"image": image, "annotations": annotation}
            for image, annotation in zip(images, annotations)
        ]
        self._samples = samples

    def _get_all_images(self) -> List[Path]:
        images = list(self._original_image_path.glob("*.png"))
        images.extend(self._original_image_path.glob("*.jpg"))
        images.extend(self._original_image_path.glob("*.jpeg"))
        return images

    def _generate_split_indices(self):
        if self._is_shuffled_splits:
            self._generate_shuffled_indices()
        else:
            self._generate_indices()

    def _generate_shuffled_indices(self):
        n_validation_samples = self.n_samples - self._n_train_samples
        sampling_frequency = self.n_samples // n_validation_samples
        self._i_val = [x * sampling_frequency for x in range(n_validation_samples)]
        self._i_train = list({x for x in range(self.n_samples)} - set(self._i_val))

    def _generate_indices(self):
        self._i_train = [x for x in range(self._n_train_samples)]
        self._i_val = [x for x in range(self._n_train_samples, self.n_samples)]

    def _select_samples(self):
        self._validation_samples = [self._samples[i] for i in self._i_val]
        self._training_samples = [self._samples[i] for i in self._i_train]

    def _log_successful_setup(self):
        message = f"Created dataset splits with {self.n_train_samples} train samples"
        message += f" and {self.n_validation_samples} validation samples."
        logger.info(message)


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger.info("Arguments: " + str(args))
    converter = AnnotationCoverter(args)
    converter.convert_annotations()
