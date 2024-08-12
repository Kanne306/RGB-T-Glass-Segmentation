#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import cv2
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import threading
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import queue
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch, inferences
from models import build_model

from torch.utils.tensorboard import SummaryWriter
import pydensecrf.densecrf as dcrf
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from functools import partial
import cv2
from concurrent.futures import ThreadPoolExecutor
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', default='True', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='rgbt')
    parser.add_argument('--rgbt_path', default='', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='checkpoints/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--log_dir', default='logs_rgbt/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--resume',  default='checkpoint_rgbt.pth', help='resume from checkpoint')
    parser.add_argument('--eval', default=True, action='store_true')

    # parser.add_argument('--resume', help='resume from checkpoint')
    # parser.add_argument('--eval', action='store_true')
    
    parser.add_argument('--is_rgbt', default='False', type = bool,
                        help="True for rgbt and False for rgb-only")

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser
image_queue = queue.Queue()
model = None
postprocessors = None
device = None
height, width  = 1088,1456
bridge = CvBridge()

def process_image():
    global image_queue, model, postprocessors , device, image_pub
    print("Process image")
    while rospy.is_shutdown() == False:
        if not image_queue.empty():
            image,stamp = image_queue.get()
            image = cv2.resize(image, (640, 480))
            input = image, image
            output_mask = inferences(model, postprocessors, input, device, True)
            output_mask = cv2.resize(output_mask, (width, height)).astype(np.uint8)
            pub_mask = bridge.cv2_to_imgmsg(output_mask, "mono8")
            pub_mask.header.frame_id = "camera_init"
            pub_mask.header.stamp = stamp
            image_pub.publish(pub_mask)
            print("Process stamp: ",stamp.to_sec())
        else:
            time.sleep(0.09)

def image_callback(msg):
    global image_queue, bridge
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    # print("image stamp:", msg.header.stamp.to_sec())
    stamp = msg.header.stamp
    image_queue.put([cv_image, stamp])
def main(args):
    global model, postprocessors , device, image_pub
    utils.init_distributed_mode(args)
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    device = torch.device(args.device)
    # device = torch.device('cpu')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, _, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    rospy.init_node('image_subscriber', anonymous=True)
    image_topic = "/webcam/image_raw"  # Replace with the actual image topic name
    modified_image_topic = "/TR_mask"
    rospy.Subscriber(image_topic, Image, image_callback, queue_size=2000)
    image_pub = rospy.Publisher(modified_image_topic, Image, queue_size=100)
    thread = threading.Thread(target=process_image)
    thread.start()
    rospy.spin()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('GETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    main(args)