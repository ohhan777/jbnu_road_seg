import os
import cv2
import json
import numpy as np
import glob

def create_png_labels(img_path, label_path, output_path):
    label_files = glob.glob(os.path.join(label_path, '*.json'))
    for label_file in label_files:
        img_file = os.path.join(img_path, os.path.basename(label_file).replace('.json', '.png'))
        png_label_file = os.path.join(output_path, os.path.basename(label_file).replace('.json', '.png'))
        create_png_label(img_file, label_file, png_label_file)

def read_img_file(img_file):
    img = cv2.imread(img_file)
    return img  

def read_label_file(label_file, mode):
    labels = []
    target = 'object_imcoords' if mode == 1 else 'building_imcoords' if mode == 2 else 'road_imcoords'
    with open(label_file, 'r') as f:
        data_dict = json.load(f)

    for x in data_dict['features']:
        obj = x['properties']
        label = []
        l = obj[target].split(',')
        if l[-1] == '':
            l.pop()
        l = [round(float(i)) for i in l]
        if len(l) != 0:
            label.append(l)
            label.append(int(obj['type_id']))
            labels.append(label)
        
    return labels

def create_png_label(img_file, label_file, png_label_file):
    labels = read_label_file(label_file, mode=3)
    img = cv2.imread(img_file)
    h, w, _ = img.shape
    png_label = np.zeros((h, w), dtype=np.uint8)
    
    for label in labels:
        pos = np.array(label[0]).reshape(-1, 2)
        type_id = label[1]
        png_label = cv2.fillPoly(png_label, [pos], type_id)
    
    cv2.imwrite(png_label_file, png_label)

img_path = 'data/kari-road/train/images'
label_path = 'data/kari-road/train/labels'
output_path = 'data/kari-road/train/png_labels'

os.makedirs(output_path, exist_ok=True)
create_png_labels(img_path, label_path, output_path)