#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file compute the model performance.

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""
import sys
sys.path.append("./generate_data")
import pickle
import os
import argparse
import pickle as pkl
from utils import load_path, load_data
from utils_result import compute_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_path',
        default='data',
        help='The root path to the model result directory (default: data)',
        type=str
    )
    parser.add_argument(
        '--ground_truth_path',
        default='data',
        help='The root path to the gt directory (default: data)',
        type=str
    )
    parser.add_argument(
        '--split',
        help='Path to split train/val/test to use (default: None)',
        type=str
    )
    parser.add_argument(
        '--validation',
        action='store_true',
        help='Use validation or test dataset (default: False)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Use train dataset (default: False)'
    )
    parser.add_argument(
        '--max_fiber',
        default=3,
        help='The maximum number of fiber (default: 3)',
        type=int
    )
    # Parse the arguments
    args = parser.parse_args()
    result_path = args.result_path
    ground_truth_path = args.ground_truth_path
    split = args.split
    validation = args.validation
    train = args.train
    max_fiber = args.max_fiber

    if validation and train:
        print('Choose either validation or train')
        raise NotImplementedError

    # Check if directory exist
    angles_path = os.path.join(result_path, 'angles')
    angle_gt_path = os.path.join(ground_truth_path, 'angles')
    if not os.path.exists(angles_path):
        print("Path doesn't exist: {0}".format(angles_path))
        raise NotADirectoryError
    if not os.path.exists(angle_gt_path):
        print("Path doesn't exist: {0}".format(angle_gt_path))
        raise NotADirectoryError

    # Create the data directory
    result_path = os.path.join(result_path, 'results')
    if not os.path.exists(result_path):
        print('Create new directory: {0}'.format(result_path))
        os.makedirs(result_path)
    if split:
        print('With SPLIT at: {0}'.format(split))
        with open(split, 'rb') as f:
            split = pkl.load(f)
        if train:
            split = split[0]
            prefix = 'train'
            print('TRAIN split')
        else:
            if validation:
                split = split[1]
                prefix = 'val'
                print('VAL split')
            else:
                split = split[2]
                prefix = 'test'
                print('TEST split')
    else:
        split = None
        prefix = 'full'
        print('NO split')

    # Load the data
    print('Load angle ground truth from: {0}'.format(angle_gt_path))
    angles_gt_l = load_path(angle_gt_path, 'angles')
    angles_gt = load_data(angle_gt_path, angles_gt_l, split)
    print('Angle ground truth loaded: {0}'.format(len(angles_gt)))
    print('Load angle predicted from: {0}'.format(angles_path))
    angles_predicted_l = load_path(angles_path, 'angles_predicted')
    angles_predicted = load_data(angles_path, angles_predicted_l, split)
    print('Angle prediceted loaded: {0}'.format(len(angles_predicted)))
    print('Save result under: {0}'.format(result_path))

    angular_error, success_rate,\
    over_estimated_fiber, under_estimated_fiber,\
    angle_information = compute_result(angles_predicted, angles_gt, max_f=max_fiber)

    pickle.dump(angular_error, open('{0}/{1}_{2}.pkl'.format(result_path, prefix, 'angular_error'), 'wb'))
    pickle.dump(success_rate, open('{0}/{1}_{2}.pkl'.format(result_path, prefix, 'success_rate'), 'wb'))
    pickle.dump(over_estimated_fiber, open('{0}/{1}_{2}.pkl'.format(result_path, prefix, 'over_estimated_fiber'), 'wb'))
    pickle.dump(under_estimated_fiber, open('{0}/{1}_{2}.pkl'.format(result_path, prefix, 'under_estimated_fiber'), 'wb'))
    pickle.dump(angle_information, open('{0}/{1}_{2}.pkl'.format(result_path, prefix, 'angle_information'), 'wb'))
