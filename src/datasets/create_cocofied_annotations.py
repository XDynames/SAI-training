# Merge original xml annotations into json, split training and validation
import argparse
import json
import os
import shutil

import xmltodict
from PIL import Image
from tqdm import tqdm

from detectron2.utils.logger import setup_logger


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--anno-dir",
        default="datasets/finalized_data/image_annotation",
        help="path to annotation xmls",
    )
    parser.add_argument(
        "--img-dir",
        default="datasets/finalized_data/Original_imgs",
        help="path to annotation xmls",
    )
    parser.add_argument(
        "--val-dir",
        default="datasets/stoma/val",
        help="path to validation dataset",
    )
    parser.add_argument(
        "--output",
        default="datasets/stoma/annotations",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--num-train", type=int, default=150, help="Number of training images"
    )
    parser.add_argument(
        "--shuffled-splits", action='store_true',
        help="Sets split generation mode to deterministic uniform sampling"
    )
    parser.add_argument(
        "--arabidopsis", action='store_true',
        help="Create the arabidopsis dataset"
    )
    parser.add_argument(
        "--combined", action='store_true',
        help="Create a dataset that combines barley and arabidopsis samples"
    )
    return parser


def poly2seg(polygon):
    # convert xml polygon to segmentation
    segmentation = coords2array(polygon)
    xs = segmentation[::2]
    ys = segmentation[1::2]
    xmin = min(range(len(xs)), key=xs.__getitem__)
    xmax = max(range(len(xs)), key=xs.__getitem__)
    ymin = min(range(len(ys)), key=ys.__getitem__)
    ymax = max(range(len(ys)), key=ys.__getitem__)
    y_extent = ys[ymax] - ys[ymin]
    x_extent = xs[xmax] - xs[xmin]

    if x_extent > y_extent:
        keypoints = [
            segmentation[xmin * 2],
            segmentation[xmin * 2 + 1],
            1,
            segmentation[xmax * 2],
            segmentation[xmax * 2 + 1],
            1,
        ]
    else:
        keypoints = [
            segmentation[ymin * 2],
            segmentation[ymin * 2 + 1],
            1,
            segmentation[ymax * 2],
            segmentation[ymax * 2 + 1],
            1,
        ]

    return keypoints, segmentation


def line2seg(line):
    line = coords2array(line)
    if line[0] < line[2]:
        keypoints = [line[0], line[1], 1, line[2], line[3], 1]
    else:
        keypoints = [line[2], line[3], 1, line[0], line[1], 1]
    # create a quadrilateral of width 1
    segmentation = [
        line[0],
        line[1],
        line[0] + 1,
        line[1] + 1,
        line[2] + 1,
        line[3] + 1,
        line[2],
        line[3],
    ]
    return keypoints, segmentation

def visualise_bndbox(self, anno_dict, stoma_dict):
    image_path = os.path.join(args.img_dir, anno_dict["filename"])
    img = Image.open(image_path)
    vis_pore = Visualizer(img)
    img = vis_pore.draw_box([ int(x) for x in stoma_dict['bndbox'].values()])
    img = Image.fromarray(img.get_image(),'RGB')
    img.show()
    
def bndbox2array(bndbox):
    return list(
        map(int, [bndbox["xmin"], bndbox["ymin"], bndbox["xmax"], bndbox["ymax"]])
    )


def coords2array(coords):
    return list(map(int, coords.values()))


def xyxy2xywh(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


def box_ainb(a, b):
    return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]


def catid(name):
    return 1 if name == "Open Stomata" else 0


def catid_by_pore(pore):
    return 1 if "polygon" in pore else 0


def create_annotation(stoma, pore):
    ann = {}
    bbox = bndbox2array(stoma["bndbox"])
    xywh = xyxy2xywh(bbox)
    ann["bbox"] = xywh
    ann["area"] = xywh[2] * xywh[3]
    ann["iscrowd"] = 0
    class_id = catid_by_pore(pore)

    ann["category_id"] = class_id

    if class_id == 1:
        keypoints, segmentation = poly2seg(pore["polygon"])
    else:
        keypoints, segmentation = line2seg(pore["line"])

    ann["keypoints"] = keypoints
    ann["segmentation"] = [segmentation]
    ann["num_keypoints"] = 2

    return ann


def cocofy(anno_dict):
    """ Convert annotation to COCO format.

    Returns:
      [annotation] whose elements are dictionaries with the following keys:
      area, iscrowd, bbox, category_id, segmentation, keypoints

    """

    annotations = []
    # match pores with stomata
    pores = [obj for obj in anno_dict["object"] if obj["name"] == "Stomatal Pore"]
    stomata = [obj for obj in anno_dict["object"] if obj["name"] != "Stomatal Pore"]

    # for each stoma, find the pore inside
    for stoma in stomata:
        annotation = None
        sbox = bndbox2array(stoma["bndbox"])
        for pore in pores:
            try:
                pbox = bndbox2array(pore["bndbox"])
            except:
                print("{} pore has not bbox".format(anno_dict["filename"]))

            if box_ainb(pbox, sbox):
                # check annotation
                if catid(stoma["name"]) != catid_by_pore(pore):
                    print("{} has wrong label.".format(anno_dict["filename"]))
                    print(stoma, pore)

                annotation = create_annotation(stoma, pore)

                # remove this pore from list
                pores.remove(pore)
                break

        if annotation is None:
            print(
                "{} has unmatched stoma: {}".format(
                    anno_dict["filename"], stoma["bndbox"]
                )
            )
            continue

        annotations.append(annotation)
    return annotations


def merge_annos(anno_list, img_root, anno_root, image_start=0, annotation_start=0):
    """ Merge all annotations and image information from the list into one cocofied json file.

    Args:
      ann_list ([str]): list of annotation file names
      img_root (str): image root directory
      anno_root (str): annotation root directory
      image_start (int): starting id for the images
      annotation_start (int): starting id for the annotations
    """
    images = []
    annotations = []

    # create anno dict
    for i, xml_file in enumerate(anno_list):
        image_id = image_start + i
        # get image info
        img_name = os.path.splitext(xml_file)[0] + ".png"
        with open(os.path.join(img_root, img_name), "rb") as f:
            im = Image.open(f)
        width, height = im.size

        images.append(
            {"file_name": img_name, "width": width, "height": height, "id": image_id}
        )

        with open(os.path.join(anno_root, xml_file)) as f:
            anno_dict = xmltodict.parse(f.read())["annotation"]

        assert int(anno_dict["size"]["width"]) == width
        assert int(anno_dict["size"]["height"]) == height

        annos = cocofy(anno_dict)

        for j, anno in enumerate(annos):
            anno["image_id"] = image_id
            anno["id"] = annotation_start + j

        annotation_start += len(annos)
        annotations.extend(annos)

    return images, annotations


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="data")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    anno_list = os.listdir(args.anno_dir)

    # split into train and val
    if args.shuffled_splits:
        n_validation = len(anno_list) - args.num_train
        sampling_frequency = len(anno_list) // n_validation
        i_val = [ x * sampling_frequency for x in range(n_validation) ]
        i_train = list({ x for x in range(len(anno_list)) } - set(i_val))
        train_list = [ anno_list[i] for i in i_train ] 
        val_list = [ anno_list[i] for i in i_val ]
    else:
        train_list = anno_list[: args.num_train]
        val_list = anno_list[args.num_train :]
        val_list.extend(human_trail_images)

    logger.info(
        "Creating datasets with {} train samples and {} validation samples.".format(
            len(train_list), len(val_list)
        )
    )

    logger.info("Converting image files to png:")
    # Convert jpgs to pngs
    image_list = os.listdir(args.img_dir)
    for file in tqdm(image_list):
        image_path = os.path.join(args.img_dir, file)
        new_image_path = ".".join([image_path[:-4], "png"])
        if file[-4:] == ".jpg" and not os.path.exists(new_image_path):
            img = Image.open(image_path)
            img.save(new_image_path)

    annotation_id = 0
    image_id = 0

    for i, (split_name, split_list) in enumerate(
        zip(["train", "val"], [train_list, val_list])
    ):

        split_images, split_annos = merge_annos(
            split_list, args.img_dir, args.anno_dir, image_id, annotation_id
        )

        if not os.path.exists(args.output):
            os.makedirs(args.output)

        with open(os.path.join(args.output, split_name + ".json"), "w") as f:
            json.dump(
                {
                    "images": split_images,
                    "annotations": split_annos,
                    "categories": [
                        {"id": 0, "name": "Closed"},
                        {"id": 1, "name": "Open"},
                    ],
                },
                f,
            )

        image_id += len(split_images)
        annotation_id += len(split_annos)

        if split_name == 'val':
            if not os.path.exists(args.val_dir):
                os.makedirs(args.val_dir)

            for item in split_list:
                img_name = item[:-4] + ".png"
                src = os.path.join(args.img_dir, img_name)
                dest = os.path.join(args.val_dir, img_name)
                shutil.copy(src, dest)