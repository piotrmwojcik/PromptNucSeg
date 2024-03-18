import numpy as np
import matplotlib.pyplot as plt
import os
import json

def read_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


test_path_raw = 'datasets/pannuke/Images/'
run_name = '_nl'
prompt_path = f'prompts/pannuke123{run_name}/'
to_save = f'results/prompts/pannuke123/{run_name}/'
os.makedirs(to_save, exist_ok=True)

gt = read_from_json('../prompter/datasets/pannuke123/test.json')


def make_array(file, gt):
    points_list = []
    categories = gt[file]
    for category_index, coordinates_list in enumerate(categories):
        for coordinates in coordinates_list:
            # Append [x, y, category] to the points_list
            points_list.append(coordinates + [category_index])

    return np.array(points_list)

brighter_colormap = np.array([
        '#FF6666',  # Brighter red
        '#6666FF',  # Brighter blue
        '#66FF66',  # Brighter green
        '#CD853F',  # Brighter brown
        '#FFFF66',  # Brighter yellow
        '#FF00FF',  # Brighter magenta
        '#00FFFF'  # Brighter cyan
    ])
colormap = np.array(['red', 'blue', 'green', 'brown', 'yellow', 'black'])

i=0
for file in sorted(os.listdir(test_path_raw)):
    if file[0] != '3' or file[-3:] != 'png' or i > 50:
        continue

    i += 1
    print(f'{i}. {test_path_raw + file}')

    im = plt.imread(test_path_raw + file)
    implot = plt.imshow(im, cmap='gray')


    gt_prompt = make_array(test_path_raw + file, gt)
    if len(gt_prompt) > 0:
        categories = gt_prompt[:, 2]
        plt.scatter(gt_prompt[:, 0], gt_prompt[:, 1], s=40, c=brighter_colormap[categories.astype(int)], marker='o')



    a = np.load(prompt_path + file[:-4] + '.npy')
    if len(a) > 0:
        categories = a[:, 2]
        plt.scatter(a[:, 0], a[:, 1], s=40, c=colormap[categories.astype(int)], marker='+')

    plt.savefig(to_save + file)
    plt.clf()

