import os

from detectron2.data.datasets.register_coco import register_coco_instances

# ==== datasets and splits for Stomata detection ====

_SPLITS_STOMA_DETECTION = {
    "stoma_detection_train": (
        "stoma_detection/train",
        "stoma_detection/annotations/train.json",
    ),
    "stoma_detection_val": (
        "stoma_detection/val",
        "stoma_detection/annotations/val.json",
    ),
}

_STOMA_DETECTION_META = {"thing_classes": ["Stomata"]}


_SPLITS_STOMA = {
    "stoma_train": ("stoma/images", "stoma/annotations/train.json"),
    "stoma_val": ("stoma/images", "stoma/annotations/val.json"),
}

_STOMA_META = {
    "thing_classes": ["Closed", "Open"],
    "keypoint_names": ("left", "right"),
    "keypoint_flip_map": (("left", "right"),),
    "keypoint_connection_rules": [("left", "right", (102, 204, 255))],
}


def register_stoma(root):
    # coco style datasets
    for key, (image_root, json_file) in _SPLITS_STOMA_DETECTION.items():
        register_coco_instances(
            key,
            _STOMA_DETECTION_META,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

    for key, (image_root, json_file) in _SPLITS_STOMA.items():
        register_coco_instances(
            key,
            _STOMA_META,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# ==== datasets and splits for Stomata measurements ====
# TODO: add datasets and splits for Stomata measurements here


# register_stoma("datasets")
