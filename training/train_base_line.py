import torch
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
import wandb
print(wandb.__version__)
wandb.login()
# Necessary/extra dependencies. 
import os
import gc
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

TRAIN_PATH = '../data/images/'
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 10

print(f'Number of extra images: {len(os.listdir(TRAIN_PATH))}') 

# Load image level csv file
extra_df = pd.read_csv('../data/image_labels.csv')
print('Number of ground truth bounding boxes: ', len(extra_df))

# Number of unique labels
label_to_id = {label: i for i, label in enumerate(extra_df.label.unique())}
print('Unique labels: ', label_to_id)

# Group together bbox coordinates belonging to the same image. 
image_bbox_label = {} # key is the name of the image, value is a dataframe with label and bbox coordinates. 
for image, df in extra_df.groupby('image'): 
    image_bbox_label[image] = df.reset_index(drop=True)

# Visualize
extra_df.head(5)

# Create train and validation split.
train_names, valid_names = train_test_split(list(image_bbox_label), test_size=0.2, random_state=42)
print(f'Size of dataset: {len(image_bbox_label)},\
       training images: {len(train_names)},\
       validation images: {len(valid_names)}')

os.makedirs('tmp/nfl_extra/images/train', exist_ok=True)
os.makedirs('tmp/nfl_extra/images/valid', exist_ok=True)

os.makedirs('tmp/nfl_extra/labels/train', exist_ok=True)
os.makedirs('tmp/nfl_extra/labels/valid', exist_ok=True)

# Move the images to relevant split folder.
for img_name in tqdm(train_names):
    copyfile(f'{TRAIN_PATH}/{img_name}', f'tmp/nfl_extra/images/train/{img_name}')

for img_name in tqdm(valid_names):
    copyfile(f'{TRAIN_PATH}/{img_name}', f'tmp/nfl_extra/images/valid/{img_name}')

# Create .yaml file 
import yaml

data_yaml = dict(
    train = 'tmp/nfl_extra/images/train',
    val = 'tmp/nfl_extra/images/valid',
    nc = 5,
    names = list(extra_df.label.unique())
)

# Note that I am creating the file in the yolov5/data/ directory.
with open('tmp/nfl_extra/yolov5/data/data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=True)
    

def get_yolo_format_bbox(img_w, img_h, box):
    """
    Convert the bounding boxes in YOLO format.
    
    Input:
    img_w - Original/Scaled image width
    img_h - Original/Scaled image height
    box - Bounding box coordinates in the format, "left, width, top, height"
    
    Output:
    Return YOLO formatted bounding box coordinates, "x_center y_center width height".
    """
    w = box.width # width 
    h = box.height # height
    xc = box.left + int(np.round(w/2)) # xmin + width/2
    yc = box.top + int(np.round(h/2)) # ymin + height/2

    return [xc/img_w, yc/img_h, w/img_w, h/img_h] # x_center y_center width height
    
# Iterate over each image and write the labels and bbox coordinates to a .txt file. 
for img_name, df in tqdm(image_bbox_label.items()):
    # open image file to get the height and width 
    img = cv2.imread(TRAIN_PATH+'/'+img_name)
    height, width, _ = img.shape 
    
    # iterate over bounding box df
    bboxes = []
    for i in range(len(df)):
        # get a row
        box = df.loc[i]
        # get bbox in YOLO format
        box = get_yolo_format_bbox(width, height, box)
        bboxes.append(box)
    
    if img_name in train_names:
        img_name = img_name[:-4]
        file_name = f'tmp/nfl_extra/labels/train/{img_name}.txt'
    elif img_name in valid_names:
        img_name = img_name[:-4]
        file_name = f'tmp/nfl_extra/labels/valid/{img_name}.txt'
        
    with open(file_name, 'w') as f:
        for i, bbox in enumerate(bboxes):
            label = label_to_id[df.loc[i].label]
            bbox = [label]+bbox
            bbox = [str(i) for i in bbox]
            bbox = ' '.join(bbox)
            f.write(bbox)
            f.write('\n')

# Extra utility function that can be used for inference. 
def convert_yolo_bbox(img_w, img_h, box):
    """
    Input:
    img_w - Original/Scaled image width
    img_h - Original/Scaled image height
    box - YOLO formatted bbox coordinates in the format, "x_center, y_center, width, height"
    
    Output:
    Return bounding box coordinates in the format, "left, width, top, height"
    """
    xc, yc = int(np.round(box[0]*img_w)), int(np.round(box[1]*img_h))
    w, h = int(np.round(box[2]*img_w)), int(np.round(box[3]*img_h))

    left = xc - int(np.round(w/2))
    top = yc - int(np.round(h/2))

    return [left, top, w, h]