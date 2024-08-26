import os

from detectron2.data.datasets.register_coco import register_coco_instances

# ==== datasets and splits for Stomata detection ====

_SPLITS_STOMA_DETECTION = {
    "stomata_barley_train": (
        "stoma/images",
        "stoma/annotations/train.json",
    ),
    "stomata_barley_val": (
        "stoma/images",
        "stoma/annotations/val.json",
    ),
}

_STOMA_DETECTION_META = {
    "thing_classes": [
        "Closed Stomata",
        "Open Stomata",
        "Stomatal Pore",
        "Subsidiary cells",
    ],
    "keypoint_names": ("left", "right"),
    "keypoint_flip_map": (("left", "right"),),
    "keypoint_connection_rules": [("left", "right", (102, 204, 255))],
}


_SPLITS_STOMA = {
    "barley_train": ("stoma/images", "stoma/annotations/train.json"),
    "barley_val": ("stoma/images", "stoma/annotations/val.json"),
}

_STOMA_META = {
    "thing_classes": [
        "Closed Stomata",
        "Open Stomata",
        "Stomatal Pore",
        "Subsidiary cells",
    ],
    "keypoint_names": ("left", "right"),
    "keypoint_flip_map": (("left", "right"),),
    "keypoint_connection_rules": [("left", "right", (102, 204, 255))],
}


def register_stomata_dataset(root):
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
