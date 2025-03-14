import os

# # Get the current working directory (the directory of the notebook)
# notebook_dir = os.path.dirname(os.path.abspath("__file__"))

# # Change the current working directory to the parent directory
# parent_dir = os.path.abspath(os.path.join(notebook_dir, os.pardir))
# os.chdir(parent_dir)

print("Current working directory:", os.getcwd())

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import numpy as np

# # Define a dummy args class to simulate the arguments
# class Args:
#     img_chexpert_file = '/data/Datasets/mimic_cxr_processed/mimic-cxr-chexpert.csv'
#     nvidia_bounding_box_file = '/data/Datasets/mimic_cxr_processed/mimic-cxr-annotation.csv'
#     imagenome_bounding_box_file = 'path/to/imagenome_bounding_box_file/'
#     imagenome_radgraph_landmark_mapping_file = 'path/to/imagenome_radgraph_landmark_mapping_file.json'
#     mini_data = 100
#     chexpert_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',\
# 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',\
# 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
#     landmark_names_spec = ['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',\
#                         'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',\
#                         'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',\
#                         'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',\
#                         'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',\
#                         'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',\
#                         'cardiophrenic_sulcus', 'mediastinal', 'spine', 'rib', 'right_atrium', 'right_ventricle', 'aorta', 'svc',\
#                         'interstitium', 'parenchymal', 'cavoatrial_junction', 'stomach', 'clavicle']
#     full_anatomy_names = ['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',\
#                             'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',\
#                             'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',\
#                             'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',\
#                             'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',\
#                             'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',\
#                             'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',\
#                             'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other']

# # Create an instance of the Args class
# args = Args()
import argparse

def get_args(arg_list=None):
     parser = argparse.ArgumentParser(description='AGXNet Training on MIMIC-CXR dataset.')

     # Experiment
     parser.add_argument('--exp-dir', metavar='DIR', default='/PROJECT DIR/EXPERIMENT DIR/EXPERIMENT NAME', help='experiment path')
     parser.add_argument("--image-path-ocean-shared", default='/data/Datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files/',
                         help="path to the image files")
     # Dataset
     parser.add_argument('--img-chexpert-file', metavar='PATH', default='/data/Datasets/mimic_cxr_processed/mimic-cxr-chexpert.csv',
                         help='master table including the image path and chexpert labels.')
     parser.add_argument('--radgraph-sids-npy-file', metavar='PATH', default='/data/Datasets/mimic_cxr_processed/landmark_observation_sids.npy',
                         help='radgraph study ids.')
     parser.add_argument('--radgraph-adj-mtx-npy-file', metavar='PATH', default='/data/Datasets/mimic_cxr_processed/landmark_observation_adj_mtx.npy',
                         help='radgraph adjacent matrix landmark - observation.')
     parser.add_argument('--nvidia-bounding-box-file', metavar='PATH', default='/data/Datasets/mimic_cxr_processed/mimic-cxr-annotation.csv',
                         help='bounding boxes annotated for pneumonia and pneumothorax.')
     parser.add_argument('--imagenome-bounding-box-file', metavar='PATH',
                    default='/data/Datasets/mimic_cxr_processed/',
                    help='ImaGenome bounding boxes for 21 landmarks.')
     parser.add_argument('--chexpert-names', nargs='+', default=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
     'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
     'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'])
     parser.add_argument('--full-anatomy-names', nargs='+', default=['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
     'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
     'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
     'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
     'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
     'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
     'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
     'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other'])
     parser.add_argument('--landmark-names-spec', nargs='+', default=['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
     'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
     'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
     'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
     'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
     'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
     'cardiophrenic_sulcus', 'mediastinal', 'spine', 'rib', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
     'interstitium', 'parenchymal', 'cavoatrial_junction', 'stomach', 'clavicle'])
     parser.add_argument('--landmark-names-unspec', nargs='+', default=['cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other'])
     parser.add_argument('--full-obs', nargs='+', default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
     'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
     'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
     'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
     'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
     'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
     'tail_abnorm_obs', 'excluded_obs'])
     parser.add_argument('--norm-obs', nargs='+', default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free', 'expand', 'hyperinflate'])
     parser.add_argument('--abnorm-obs', nargs='+', default=['effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
     'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
     'device', 'engorgement', 'picc', 'clip', 'elevation', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
     'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
     'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration'])
     parser.add_argument('--tail-abnorm-obs', nargs='+', default=['tail_abnorm_obs'])
     parser.add_argument('--excluded-obs', nargs='+', default=['excluded_obs'])
     parser.add_argument('--selected-obs', nargs='+', default=['pneumothorax'])
     parser.add_argument('--labels', nargs='+',
                         default=['0 (No Pneumothorax)', '1 (Pneumothorax)'])

     # PNU labels
     parser.add_argument('--warm-up', default=2, type=int,
                         help='number of epochs warm up before PU learning.')

     # Training
     parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',
                         help='PyTorch image models')
     parser.add_argument('--pool1', metavar='ARCH', default='average',
                         help='type of pooling layer for net1. the options are: average, max, log-sum-exp')
     parser.add_argument('--pool2', metavar='ARCH', default='average',
                         help='type of pooling layer for net2. the options are: average, max, log-sum-exp')
     parser.add_argument('--gamma', default=5.0, type=float,
                         help='hyper-parameter for log-sum-exp pooling layer')
     parser.add_argument('--nets-dep', metavar='ARCH', default='dep',
                         help='whether pass CAM1 to net2, dep or indep')
     parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                         help='number of data loading workers (default: 4)')
     parser.add_argument('--epochs', default=30, type=int, metavar='N',
                         help='number of total epochs to run')
     parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
     parser.add_argument('-b', '--batch-size', default=8, type=int,
                         metavar='N',
                         help='mini-batch size (default: 256), this is the total '
                              'batch size of all GPUs on the current node when '
                              'using Data Parallel or Distributed Data Parallel')
     parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                         metavar='LR', help='initial learning rate', dest='lr')
     parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum')
     parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                         metavar='W', help='weight decay (default: 1e-4)',
                         dest='weight_decay')
     parser.add_argument('--training-schedule', default='end-to-end',
                         help='training schedule. the options are: '
                              '(1) end-to-end: concurrently train two networks end-to-end; '
                              '(2) interleave: train two networks independently')
     parser.add_argument('--loss1', default='BCE_W',
                         help='anatomical landmark loss type.')
     parser.add_argument('--loss2', default='CE',
                         help='observation loss type.')
     parser.add_argument('--beta', default=0.1, type=float,
                         help='scaling weight of CAM1')
     parser.add_argument('--dropout-method', default='random',
                         help='dropout method. 1, random, 2 proportional')
     parser.add_argument('--dropout-rate', default=0.1, type=float,
                         help='randomly drop out x% of channels in the last conv. layer of net1')
     parser.add_argument('--cam1-norm', default='norm',
                         help='cam1 normalization method. default: norm [0, 1]')
     parser.add_argument('--cam2-norm', default='norm',
                         help='cam2 normalization method. default: norm [0, 1]')
     parser.add_argument('-p', '--print-freq', default=10, type=int,
                         metavar='N', help='print frequency (default: 10)')
     parser.add_argument('--anneal-function', default='constant', type=str,
                         help='possible anneal functions: (1) logistic, (2) linear, (3) constant.')
     parser.add_argument('--k',  default=0.5, type=float,
                         help='rate of annealing function.')
     parser.add_argument('--x0', default=15, type=int,
                         help='offset of annealing function.')
     parser.add_argument('--C', default=1.0, type=float,
                         metavar='W', help='weight of observation classification loss.')
     parser.add_argument('--resume', default='', type=str, metavar='PATH',
                         help='path to latest checkpoint (default: none)')
     parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                         help='use pre-trained model')
     parser.add_argument('--world-size', default=1, type=int,
                         help='number of nodes for distributed training')
     parser.add_argument('--rank', default=0, type=int,
                         help='node rank for distributed training')
     parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                         help='url used to set up distributed training')
     parser.add_argument('--dist-backend', default='nccl', type=str,
                         help='distributed backend')
     parser.add_argument('--seed', default=None, type=int,
                         help='seed for initializing training. ')
     parser.add_argument('--gpu', default=0, type=int,
                         help='GPU id to use.')
     parser.add_argument('--multiprocessing-distributed', action='store_true',
                         help='Use multi-processing distributed training to launch '
                              'N processes per node, which has N GPUs. This is the '
                              'fastest way to use PyTorch for either single node or '
                              'multi node data parallel training')
     parser.add_argument('--ngpus-per-node', default=2, type=int,
                         help='Number of GPUs per node.')
     # Image Augmentation
     parser.add_argument('--resize', default=512, type=int,
                         help='input image resize')
     parser.add_argument('--crop', default=448, type=int,
                         help='resize image crop')
     parser.add_argument('--mini-data', default=None, type=int, help='small dataset for debugging')
     args = parser.parse_args(arg_list)
     args.N_landmarks_spec = len(args.landmark_names_spec)
     args.N_selected_obs = len(args.selected_obs)
     args.N_labels = len(args.labels)
     return args

# Dummy data for radgraph_sids and radgraph_adj_mtx


def create_dataset(args, mode='train', model_type = 'bb'):
     radgraph_sids= np.load("/data/Datasets/mimic_cxr_processed/landmark_observation_sids_v2.npy")
     radgraph_adj_mtx = np.load("/data/Datasets/mimic_cxr_processed/landmark_observation_adj_mtx_v2.npy")

     # Assuming the MIMICCXRDataset class is defined in the same notebook or imported from another module
     from datasets.dataset_mimic_cxr import MIMICCXRDataset

     # Create an instance of the dataset
     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
     dataset = MIMICCXRDataset(args, radgraph_sids, radgraph_adj_mtx, mode=mode,model_type=model_type, transform=transforms.Compose([
                                   transforms.Resize(args.resize),
                                   transforms.CenterCrop(args.resize),
                                   transforms.ToTensor(),
                                   normalize
                              ])
                              )
     return dataset




def main():
    args = get_args()
    dataset = create_dataset(args)

    # Create a DataLoader to iterate through the dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate through the DataLoader
    for batch in dataloader:
        print(batch)
        break

if __name__ == "__main__":
    main()