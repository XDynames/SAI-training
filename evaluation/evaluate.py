import os
import json
import argparse

import matplotlib.pyplot as plt

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        help="Path to folder containing ground truth prediction pair .json files",
        required=True
    )
    parser.add_argument(
        "--csv_output",
        help="Where to save .csv version of data",
        default=None
    )
    return parser.parse_args()

def load_gt_pred_pairs(directory):
    files = os.listdir(directory)
    image_detections = []
    for file in files:
        if not file[-4:] == 'json': continue 
        with open(os.path.join(directory,file), "r") as read_file:
            image_data = json.load(read_file)
            image_detections.append(image_data)
    return image_detections

def associate_ids_to_pars(image_dicts):
    stoma_properties = dict()
    for image_dict in image_dicts:
        detections = image_dict['detections']
        for detection in detections:
            gt, pred = detection['gt'], detection['pred']
            property_pairs = {
                'width' : { 'gt' : gt['width'], 'pred' : pred['width'] },
                'length' : { 'gt' : gt['length'], 'pred' : pred['length'] },
                'area' : { 'gt' : gt['area'], 'pred' : pred['area'] },
                'class' : { 'gt' : gt['category_id'], 'pred' : pred['class'] },
                'confidence' : pred['confidence'],
            }
            stoma_properties[gt['id']] = property_pairs
    return stoma_properties

def extract_property(stoma_id_dict, key): 
    property_pairs = []
    for stoma_id in stoma_id_dict.keys():
        property_pair = stoma_id_dict[stoma_id]
        gt_value = property_pair[key]['gt']
        pred_value = property_pair[key]['pred']
        property_pairs.append([gt_value, pred_value])
    return property_pairs

def write_to_csv(stoma_id_dict, filepath): 
    column_names = [
        'id','class','pred_class','length','pred_length',
        'width','pred_width','area','pred_area','confidence'
    ]
    csv = ','.join(column_names) + '\n'
    for key in stoma_id_dict.keys():
        values = [key]
        detection = stoma_id_dict[key]
        for stoma_property in detection.keys():
            if stoma_property == 'confidence':
                values.append(detection[stoma_property])
            else:
                values.extend([
                    detection[stoma_property]['gt'],
                    detection[stoma_property]['pred']
                ])
        values = [ str(x) for x in values ]
        csv += ','.join(values) + '\n'

    with open(filepath, "w") as file:
        file.write(csv)

def plot_scatter(data, line_range, title):
    gt = [ point[0] for point in data ]
    pred = [ point[1] for point in data ]
    x = [ i / 2 for i in range(line_range[0],line_range[1]) ]
    y = x
    plt.plot(x, y, label="y = x")
    plt.title(title)
    plt.xlabel("Observed (pixels)")
    plt.ylabel("Predicted (pixels)")
    plt.scatter(gt, pred, label="RCNN v2")
    plt.legend(loc='lower right')
    plt.show()

if __name__=='__main__':
    # Get Commandline arguments
    args = get_arguments()
    # Load predictions and GT from .jsons
    gt_pred_dicts = load_gt_pred_pairs(args.directory)
    # Extract pairs into stomal id lists
    stoma_id_pairs = associate_ids_to_pars(gt_pred_dicts)
    # Extract individual property pairs
    width_pairs = extract_property(stoma_id_pairs, 'width')
    length_pairs = extract_property(stoma_id_pairs, 'length')
    class_pairs = extract_property(stoma_id_pairs, 'class')
    area_pairs = extract_property(stoma_id_pairs, 'area')
    # Covert to csv formatted file
    if not args.csv_output == None:
        write_to_csv(stoma_id_pairs, args.csv_output)
    # Display Plots
    plot_scatter(width_pairs, [0,140], "Width Predictions")
    plot_scatter(length_pairs, [160,400], "Length Predictions")
    plot_scatter(area_pairs, [0,17000], "Area Predictions")


