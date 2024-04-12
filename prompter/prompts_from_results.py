import json

import torch
import pickle
import numpy as np
from utils import *

json_file = open('name_to_id.json')
file_dict = json.load(json_file)

with open('results.pkl', 'rb') as f:
    data = pickle.load(f)

path_to_save = '../segmentor/prompts/pannuke123_detr_035/'
mkdir(path_to_save)

for name, key in file_dict.items():
    print(name, key)

    pd_scores = np.array(data[key][0].cpu())
    above_treshold = pd_scores > 0.35

    pd_classes = np.array(data[key][2].cpu())[above_treshold] - 1

    bboxes = np.array(data[key][1].cpu())[above_treshold]

    pd_points = np.empty((bboxes.shape[0], 3), dtype=np.float32)
    pd_points[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
    pd_points[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
    pd_points[:, 2] = pd_classes

    filename = f'{path_to_save}2_{name[6:-4]}.npy'
    np.save(filename, pd_points)
