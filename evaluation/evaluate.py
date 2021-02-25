import os
import json
import argparse

import matplotlib.pyplot as plt

from legacy_annotation_parser import read_legacy_val

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        help="Path to folder containing ground truth prediction pair .json files",
        required=True
    )
    parser.add_argument(
        "--csv-output",
        help="Where to save .csv version of data",
        default=None
    )
    parser.add_argument(
        "--legacy-csv",
        help="Predictions from legacy RCNN model",
        default=None
    )
    parser.add_argument(
        "--plot",
        help="Plot results",
        action='store_true'
    )
    return parser.parse_args()

def load_gt_pred_pairs(directory):
    files = os.listdir(directory)
    image_detections = []
    for file in files:
        if not file[-4:] == 'json': continue 
        with open(os.path.join(directory,file), "r") as read_file:
            image_data = json.load(read_file)
            image_data['image_name'] = file[:-4].split('-')[0]
            image_detections.append(image_data)
    return image_detections

def associate_ids_to_pairs(image_dicts):
    stoma_properties = dict()
    counter = 0
    for image_dict in image_dicts:
        image_name = image_dict['image_name']
        detections = image_dict['detections']
        for detection in detections:
            d_width, d_length, d_area, d_class, d_conf = {}, {}, {}, {}, {}
            if 'gt' in detection:
                gt = detection['gt']
                pred = detection['pred']
                d_width['gt'] = gt['width']
                d_length['gt'] = gt['length']
                d_area['gt'] = gt['area']
                d_class['gt'] = gt['category_id']
            else:
                pred = detection
            d_width['pred'] = pred['width']
            d_length['pred'] = pred['length']
            d_area['pred'] = pred['area']
            d_class['pred'] = pred['category_id']
            d_conf['pred'] = pred['confidence']
            d_image_name = {'pred': image_name}

            property_pairs = {
                'width' : d_width,
                'length' : d_length,
                'area' : d_area,
                'class' : d_class,
                'confidence' : d_conf,
                'image_name' : d_image_name,
            }
            if 'gt' in detection:
                stoma_properties[gt['id']] = property_pairs
            else:
                stoma_properties[counter] = property_pairs
                counter += 1
    return stoma_properties

def extract_property(stoma_id_dict, key):
    property_pairs = []
    for stoma_id in stoma_id_dict.keys():
        property_pair = stoma_id_dict[stoma_id]
        pred_value = property_pair[key]['pred']
        if 'gt' in property_pair[key]:
            gt_value = property_pair[key]['gt']
            property_pairs.append([gt_value, pred_value])
        else:
            property_pairs.append([pred_value])
    return property_pairs

def write_to_csv(stoma_id_dict, filepath):
    if 'gt' in stoma_id_dict[list(stoma_id_dict.keys())[0]]['class']:
        column_names = [
            'id', 'image_name', 'class','pred_class','length',
            'pred_length', 'width','pred_width','area',
            'pred_area','confidence',
        ]
    else:
        column_names = [
            'id', 'image_name', 'pred_class','pred_length',
            'pred_width','pred_area','confidence'
        ]
 
    colums_keys = [
        'image_name', 'class', 'length', 'width', 'area', 'confidence'
    ]
    csv = ','.join(column_names) + '\n'
    for key in stoma_id_dict.keys():
        values = [key]
        detection = stoma_id_dict[key]
        for stoma_property in colums_keys:
            values.append(detection[stoma_property]['pred'])
            if detection[stoma_property] is dict:
                values.append(detection[stoma_property]['gt'])
        values = [ str(x) for x in values ]
        csv += ','.join(values) + '\n'

    with open(filepath, "w") as file:
        file.write(csv)

def plot_x_y_line(line_range, title):
    x = [ i / 2 for i in range(line_range[0],line_range[1]) ]
    y = x
    plt.plot(x, y, label="y = x")
    plt.title(title)

def plot_scatter(data, label, colour):
    gt = [ point[0] for point in data ]
    pred = [ point[1] for point in data ]
    plt.xlabel("Observed (pixels)")
    plt.ylabel("Predicted (pixels)")
    plt.scatter(gt, pred, label=label, c=colour)
    plt.legend(loc='lower right')

if __name__=='__main__':
    # Get Commandline arguments
    args = get_arguments()
    # Load predictions and GT from .jsons
    gt_pred_dicts = load_gt_pred_pairs(args.directory)
    # Extract pairs into stomal id lists
    stoma_id_pairs = associate_ids_to_pairs(gt_pred_dicts)
    # Extract individual property pairs
    width_pairs = extract_property(stoma_id_pairs, 'width')
    length_pairs = extract_property(stoma_id_pairs, 'length')
    class_pairs = extract_property(stoma_id_pairs, 'class')
    area_pairs = extract_property(stoma_id_pairs, 'area')
    # Covert to csv formatted file
    if not args.csv_output is None:
        write_to_csv(stoma_id_pairs, args.csv_output)
    if not args.legacy_csv is None:
        files = os.listdir(args.legacy_csv)
        legacy_all_preds = []
        for file in files:
            # Loads in area, length and width in that order
            legacy_open_closed_paris = read_legacy_val(os.path.join(args.legacy_csv, file))
            legacy_all_pairs = [ x for y in legacy_open_closed_paris for x in y ]
            legacy_all_preds.append(legacy_all_pairs)
        print(legacy_all_preds)

    # Display Plots
    if args.plot:
        plot_x_y_line([0,140], "Width Predictions")
        plot_scatter(width_pairs, "RCNN v2", '#2ca02c')
        if not args.legacy_csv is None:
            plot_scatter(legacy_all_preds[2], "Legacy RCNN", '#ff7f0e')
        plt.show()
        plot_x_y_line([160,400], "Length Predictions")
        plot_scatter(length_pairs, "RCNN v2", '#2ca02c')
        if not args.legacy_csv is None:
            plot_scatter(legacy_all_preds[1], "Legacy RCNN", '#ff7f0e')
        plt.show()
        plot_x_y_line([0,17000], "Area Predictions")
        plot_scatter(area_pairs, "RCNN v2", '#2ca02c')
        if not args.legacy_csv is None:
            plot_scatter(legacy_all_preds[0], "Legacy RCNN", '#ff7f0e')
        plt.show()

