from typing import Dict, List

from loguru import logger

from arguments import get_parser
from create_annotations import AnnotationConverter
from constants import DINOLIGHT_NAMES_TO_CATEGORY_ID


class DinolightAnnotationConverter(AnnotationConverter):
    @property
    def _names_to_category_ids(self) -> Dict:
        return DINOLIGHT_NAMES_TO_CATEGORY_ID

    def _create_annotations(self):
        self._create_open_stomata_annotations()
        self._create_closed_stomata_annotations()

    def _create_open_stomata_annotations(self):
        for annotation in self._annotations_by_type["Open Stomata"]:
            pore_annotation = self._maybe_find_pore_of_stomata(annotation)
            if pore_annotation is not None:
                self._add_open_stomata_to_dataset(annotation, pore_annotation)

    def _add_open_stomata_to_dataset(self, stomata: Dict, pore: Dict):
        xyxy_bbox = self._bbox_dict_to_xyxy(stomata["bndbox"])
        bounding_box = self._xyxy_to_xywh(xyxy_bbox)
        keypoints = self._get_open_stomata_keypoints(pore["polygon"])
        converted_annotation = {
            "bbox": bounding_box,
            "area": self._get_bbox_area(bounding_box),
            "iscrowd": 0,
            "category_id": self._names_to_category_ids["Open Stomata"],
            "segmentation": [self._annotation_to_polygon(pore)],
            "num_keypoints": 2,
            "keypoints": keypoints,
            "image_id": self._image_id,
            "id": self._annotation_id,
        }
        self._annotations.append(converted_annotation)
        self._annotation_id += 1

    def _create_closed_stomata_annotations(self):
        for annotation in self._annotations_by_type["Closed Stomata"]:
            pore_annotation = self._maybe_find_pore_of_stomata(annotation)
            if pore_annotation is not None:
                self._add_closed_stomata_to_dataset(annotation, pore_annotation)

    def _add_closed_stomata_to_dataset(self, stomata: Dict, pore: Dict):
        xyxy_bbox = self._bbox_dict_to_xyxy(stomata["bndbox"])
        bounding_box = self._xyxy_to_xywh(xyxy_bbox)
        keypoints = self._get_closed_stomata_keypoints(pore["line"])
        segmentation = self._get_mask_from_keypoints(keypoints)
        converted_annotation = {
            "bbox": bounding_box,
            "area": self._get_bbox_area(bounding_box),
            "iscrowd": 0,
            "category_id": self._names_to_category_ids["Closed Stomata"],
            "segmentation": segmentation,
            "num_keypoints": 2,
            "keypoints": keypoints,
            "image_id": self._image_id,
            "id": self._annotation_id,
        }
        self._annotations.append(converted_annotation)
        self._annotation_id += 1

    def _get_mask_from_keypoints(self, keypoints: List[float]) -> List[List[float]]:
        polygon = [
            keypoints[0],
            keypoints[1],
            keypoints[3],
            keypoints[4],
            keypoints[3] + 1.0,
            keypoints[4] + 1.0,
            keypoints[0] + 1.0,
            keypoints[1] + 1.0,
        ]
        return [polygon]


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger.info("Arguments: " + str(args))
    converter = DinolightAnnotationConverter(args)
    converter.convert_annotations()
