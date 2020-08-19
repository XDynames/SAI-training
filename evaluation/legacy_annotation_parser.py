'''
	Parsers for old annotation formats:
		- Keypoint Regression (read_regression_predictions())
		- Original RCNN (read_legacy() / read_legacy_val())
'''

import os, csv

# Parses the provided annotation file and creates a dictionary
# associating images with their keyppoint labels
def read_regression_predictions(filename):
	target_dict = dict()

	with open(filename, 'r') as file:
		for line in file.readlines():
			tokens = line.split()
			target_dict[tokens[0]] = {
				"Open" : True if tokens[0].split('_')[0] == 'open' else False,
				"A" : [ float(tokens[2][1:-2]), float(tokens[3][:-2]) ],
				"B" : [ float(tokens[5][1:-2]), float(tokens[6][:-2]) ],
				"AB": None,
				#"C" : [ float(tokens[8][1:-2]), float(tokens[9][:-2]) ],
				#"D" : [ float(tokens[11][1:-2]), float(tokens[12][:-2]) ] 
				}
	return target_dict


def read_legacy(filename):
	with open(filename) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		next(csv_reader) # Skip column names
		# List of [ prediction, GT ] pairs
		prediction_gt = [ [float(line[1]), float(line[2])] for line in csv_reader ]
	
	return prediction_gt

# Only returns predictions for validation set samples
def read_legacy_val(filename):
	with open(filename) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		next(csv_reader) # Skip column names
		# List of [ prediction, GT ] pairs
		annotations = { line[0] : {"AB_pred_gt" : [float(line[1]), float(line[2])],
								   "Open" : True if line[3] == "Open" else False, 
								   } for line in csv_reader }

	open_val_pred_gt, closed_val_pred_gt = [], []
	for image_name in annotations.keys():
		if int(image_name) >= 738:
			pred_gt = annotations[image_name]["AB_pred_gt"]
			if annotations[image_name]["Open"]:
				open_val_pred_gt.append(pred_gt)
			else:
				closed_val_pred_gt.append(pred_gt)

	return [open_val_pred_gt, closed_val_pred_gt]