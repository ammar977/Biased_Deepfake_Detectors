import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import pandas as pd

import os
import cv2
import numpy as np
import torch
from torch import nn, einsum
from sklearn.metrics import plot_confusion_matrix

from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from cross_efficient_vit import CrossEfficientViT
from utils import transform_frame
import glob
from os import cpu_count
import json
from multiprocessing.pool import Pool
from progress.bar import Bar
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager
from utils import custom_round, custom_video_round
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate
from transforms.albu import IsotropicResize
import yaml
import argparse
import csv


#########################
####### UTILITIES #######
#########################

def save_confusion_matrix(confusion_matrix):
  fig, ax = plt.subplots()
  im = ax.imshow(confusion_matrix, cmap="Blues")

  threshold = im.norm(confusion_matrix.max())/2.
  textcolors=("black", "white")

  ax.set_xticks(np.arange(2))
  ax.set_yticks(np.arange(2))
  ax.set_xticklabels(["original", "fake"])
  ax.set_yticklabels(["original", "fake"])
  
  ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

  for i in range(2):
      for j in range(2):
          text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", 
                         fontsize=12, color=textcolors[int(im.norm(confusion_matrix[i, j]) > threshold)])

  fig.tight_layout()
  plt.savefig(os.path.join(OUTPUT_DIR, "confusion.jpg"))
  

def save_roc_curves(correct_labels, preds, model_name, accuracy, loss, f1):
  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'k--')

  fpr, tpr, th = metrics.roc_curve(correct_labels, preds)

  model_auc = auc(fpr, tpr)


  plt.plot(fpr, tpr, label="Model_"+ model_name + ' (area = {:.3f})'.format(model_auc))

  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.savefig(os.path.join(OUTPUT_DIR, model_name +  "_" + opt.dataset + "_acc" + str(accuracy*100) + "_loss"+str(loss)+"_f1"+str(f1)+".jpg"))
  plt.clf()


def read_frames(video_path, videos):
    
    label = 0.
    

    # print(f'video path :{video_path}')
    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    frames_interval = int(frames_number / opt.frames_per_video)
    frames_paths = os.listdir(video_path)

    frames_list = []
    for index_frame, fr in enumerate(frames_paths):
        if index_frame <= 29:
            # print(f'video processed : {video_path}')
            transform = create_base_transform(config['model']['image-size'])
            image = transform(image=cv2.imread(os.path.join(video_path, fr)))['image']
            frames_list.append(image)
        else:
            break

    videos.append([video_path, frames_list])


def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

#########################
#######   MODEL   #######
#########################


# Main body
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=9, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                        help='Path to model checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='DFDC', 
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|DFDC)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--frames_per_video', type=int, default=30, 
                        help="How many equidistant frames for each video (default: 30)")
    parser.add_argument('--batch_size', type=int, default=16, 
                        help="Batch size (default: 32)")
    
    opt = parser.parse_args()
    # print(opt)
    
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
 
        
    if os.path.exists(opt.model_path):
        model = CrossEfficientViT(config=config)
        model.load_state_dict(torch.load(opt.model_path))
        model.eval()
        model = model.cuda()
    else:
        print("No model found.")
        exit()

    model_name = os.path.basename(opt.model_path)


    #########################
    ####### EXECUTION #######
    #########################

    #########################
    ####### CONSTANTS #######
    #########################
    GLOBAL_PATH_DATASET_VIDS = '/home/cosminciausu/Documents/cs523/project/data/FakeAVCeleb/RealVideo-RealAudio' #VIDS PATH == ORIGINAL DATASET PATH
    GLOBAL_PATH_DATASET_FACES = '/home/cosminciausu/Documents/cs523/project/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/FakeAVCeleb/' #EXTRACTED FACES PATH
    ammar_file = 'temp_predictions_needed.txt' # FILE TO MODIFY FOR PREDICTIONS, SEE FORMAT 

    list_vids_path = []
    list_vids_name = []

    with open(ammar_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                path = row[0].split('/')[1:]
                path[-1] = path[-1][:-4]
                path = '/'.join(path)
                # print(f'path :{path}')
                if os.path.isdir(os.path.join(GLOBAL_PATH_DATASET_FACES, path)):
                    list_vids_path.append(os.path.join(GLOBAL_PATH_DATASET_FACES, path))
                    list_vids_name.append(row[0])
                line_count += 1

    NUM_CLASSES = 1
    preds = []

    mgr = Manager()
    paths = []
    videos = mgr.list()

    folders = []
    for vid in list_vids_path:
        paths.append(vid)
    with Pool(processes=cpu_count()-1) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, videos=videos),paths):
                pbar.update()

    video_names = np.asarray([row[0] for row in videos])
    correct_test_labels = np.asarray([0 for row in videos])#[row[1] for row in videos])
    preds = []
    bar = Bar('Predicting', max=len(videos))
    nbr_vids=30
    index_vids=0
    f = open('preds_temp_ammar.txt', 'w+')#OUTPUT_DIR + '/' + CURR_POP_ID + "_" + model_name + "_labels.csv", "w+")
    f.write('video_name,pred'+'\n')
    for index, video in enumerate(videos):#videos):
        # if index_vids <nbr_vids:
        video_faces_preds = []
        video_name = video_names[index]
        faces_preds = []
        video_faces = video[1]#video[key]
        for i in range(0, len(video_faces), opt.batch_size):
            faces = video_faces[i:i+opt.batch_size]
            faces = torch.tensor(np.asarray(faces))
            if faces.shape[0] == 0:
                continue
            faces = np.transpose(faces, (0, 3, 1, 2))
            faces = faces.cuda().float()
            pred = model(faces)
            scaled_pred = []
            for idx, p in enumerate(pred):
                scaled_pred.append(torch.sigmoid(p))
            faces_preds.extend(scaled_pred)  
        current_faces_pred = sum(faces_preds)/len(faces_preds)
        face_pred = current_faces_pred.cpu().detach().numpy()[0]
        video_faces_preds.append(face_pred)
        bar.next()
        if len(video_faces_preds) > 1:
            video_pred = custom_video_round(video_faces_preds)
        else:
            video_pred = video_faces_preds[0]
        preds.append([video_pred])
        f.write(video_name + ".mp4" +  "," + str(video_pred))
            # index_vids+=1
        
        f.write("\n")
        
    f.close()
    bar.finish()
