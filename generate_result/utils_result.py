#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Util functions tp compute model performance.

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""
import sys
sys.path.append("./generate_data")
from dipy.reconst.recspeed import local_maxima, search_descending
import numpy as np
from dipy.sims.voxel import _check_directions
from utils import sh_matrix


def compute_D(grid, p):
    """Compute interpolation matrix.

    Parameters
    ----------
    grid : matrix (n, 3, float)
        The grid gradients
    p : int
        The power of the linear interpolation (limit is a nearest neighbor interpolation)
        or the spherical harmonic order
    Returns
    -------
    D : matrix (N, n, float)
        The intepolation matrix
    """
    _, SH2S_G = sh_matrix(p, grid)
    D = SH2S_G
    return D


def peak_detection_(fodf, edges, spherical_coordinates, relative_peak_threshold=0.5, sep_min=15, max_fiber=3):
    """
    Detect the peak directions from one fODF.
    :param fodf: np.array
        The fodf array for one fiber and voxel, of size res*res.
    :param relative_peak_threshold: float, 0.5
        The relative peak threshold do delete the small peaks.
    :param sep_min: float, 15
        The minimum angle between two peak (in degree)
    :param max_fiber: int, 3
        The maximum number of fiber
    :return: direction_pred: list
        A list of tuple, the directions of the peaks on the North hemisphere. The tuples are (theta, phi)
        the spherical coordinates of the peaks.
    """
    values, indices = local_maxima(fodf.flatten(), edges)
    lats, lons = spherical_coordinates
    # Get the number of peaks
    n = len(values)
    if n == 0 or (values[0] < 0.):
        # If there is no peak or all the peaks are negative, return empty array
        directions = np.array([[]])
    elif n == 1:
        # If there is only one peak (should not happen since the fODF is symmetric)
        directions = np.array([lats.flatten()[indices], lons.flatten()[indices]])
    else:
        # Remove the small peak
        odf_min = np.min(fodf.flatten())
        odf_min = odf_min if (odf_min >= 0.) else 0.
        values_norm = (values - odf_min)
        n = search_descending(values_norm, relative_peak_threshold)
        indices = indices[:n]
        directions = np.array([lats.flatten()[indices], lons.flatten()[indices]])
    # Transform in degree and rotate the peak to have all of them in the North hemisphere.
    direction = directions * 180/np.pi
    if len(direction) == 2:
        for i in range(direction.shape[1]):
            if direction[0, i] > 90:
                direction[0, i] = 180 - direction[0, i]
                direction[1, i] = (direction[1, i] - 180) %(360)
        # Delete the similar peaks.
        direction_pred = []
        indice = []

        u = direction * np.pi / 180
        colats = u[0]
        lons = u[1]
        x = np.cos(lons) * np.sin(colats)
        y = np.sin(lons) * np.sin(colats)
        z = np.cos(colats)
        vertices = np.ones((colats.size, 3))
        vertices[:, 0], vertices[:, 1], vertices[:, 2] = x.flatten(), y.flatten(), z.flatten()
        norm_vertices = np.linalg.norm(vertices, axis=-1)
        norm_vertices2 = norm_vertices[:, None].dot(norm_vertices[None, :])
        vertices = np.arccos(np.minimum(np.abs(vertices.dot(vertices.T))/norm_vertices2, 1)) * 180/np.pi
        for i in range(direction.shape[1]):
            try:
                if i not in indice:
                    direction_pred.append((direction[0, i], direction[1, i]))
                    tab = (vertices[i] < sep_min) | (180 - vertices[i] < sep_min)
                    tab[i] = False
                    indice = indice + np.arange(direction.shape[1])[tab].tolist()
            except:
                print(indice)
        if len(direction_pred) > max_fiber:
            direction_pred = direction_pred[:max_fiber]
    else:
        direction_pred = []
    return direction_pred


def compute_result(direction_pred, orientation_ground_truth,
                   fraction_tissues=None, fraction_tissues_ground_truth=None, max_f=3):
    angular_error = []
    success_rate = []

    over_estimated_fiber = []
    over_estimated_fiber_total = []

    under_estimated_fiber = []
    under_estimated_fiber_total = []

    angle_information = []

    for i in range(max_f):
        angular_error.append([])
        success_rate.append([])

        over_estimated_fiber.append([])
        over_estimated_fiber_total.append([])

        under_estimated_fiber.append([])
        under_estimated_fiber_total.append([])

        angle_information.append([])

    if fraction_tissues is not None and fraction_tissues_ground_truth is not None:
        frac_error = np.abs(fraction_tissues - fraction_tissues_ground_truth)
        print('fraction error: WM. Mean: ', np.mean(frac_error[:, 0]), ' Std: ', np.std(frac_error[:, 0]))
        print('fraction error: GM. Mean: ', np.mean(frac_error[:, 1]), ' Std: ', np.std(frac_error[:, 1]))
        print('fraction error:CSF. Mean: ', np.mean(frac_error[:, 2]), ' Std: ', np.std(frac_error[:, 2]))
    for i in range(len(direction_pred)):
        if len(orientation_ground_truth[i]) != 0:
            # Ground truth voxel i
            angle_gt = orientation_ground_truth[i]
            sticks_gt = _check_directions(angle_gt)
            if len(direction_pred[i]) != 0:
                # Prediction voxel i
                angle_pred = direction_pred[i]
                sticks_pred = _check_directions(angle_pred)
                # Prediction and ground truth angles
                norm_gt = np.linalg.norm(sticks_gt, axis=-1)
                norm_pred = np.linalg.norm(sticks_pred, axis=-1)
                norm_predgt = norm_pred[:, None].dot(norm_gt[None, :])
                angle = np.arccos(np.minimum(np.abs(sticks_pred.dot(sticks_gt.T))/norm_predgt, 1))
                # angle = np.minimum(angle, np.pi - angle)
                if np.sum(angle < 0) != 0:
                    print("ERROR")
                gt_to_pred = np.min(angle, 0)
                pred_to_gt = np.min(angle, 1)

                # Anguler error: mean angular error for each gt fiber to the closest prediction
                angular_error[len(sticks_gt) - 1].append(np.mean(gt_to_pred))

                # Success rate: voxel is well classify if each gt fiber (resp. each prediction)
                #  is near a prediction (resp. a gt)
                if len(angle_gt) == len(angle_pred):
                    if (np.sum(gt_to_pred * 180 / np.pi < 25) == len(angle_gt)) and (
                            np.sum(pred_to_gt * 180 / np.pi < 25) == len(angle_pred)):
                        success_rate[len(sticks_gt) - 1].append(1)
                    else:
                        success_rate[len(sticks_gt) - 1].append(0)
                else:
                    success_rate[len(sticks_gt) - 1].append(0)

                # False positive prediction (spurious fibers): predicted fibers which are not
                #  closer than 25° from a gt fiber
                over_estimated_fiber[len(sticks_gt) - 1].append(np.sum(pred_to_gt * 180 / np.pi >= 25))
                over_estimated_fiber_total[len(sticks_gt) - 1].append(max(0, len(pred_to_gt) - len(gt_to_pred)))
                # False negative prediction: gt fibers which are not closer than 25° from a predicted fiber
                under_estimated_fiber[len(sticks_gt) - 1].append(np.sum(gt_to_pred * 180 / np.pi >= 25))
                under_estimated_fiber_total[len(sticks_gt) - 1].append(max(0, len(gt_to_pred) - len(pred_to_gt)))

            else:
                angular_error[len(sticks_gt) - 1].append(np.pi/2)
                success_rate[len(sticks_gt) - 1].append(0)

                over_estimated_fiber[len(sticks_gt) - 1].append(0)
                over_estimated_fiber_total[len(sticks_gt) - 1].append(0)

                under_estimated_fiber[len(sticks_gt) - 1].append(len(sticks_gt))
                under_estimated_fiber_total[len(sticks_gt) - 1].append(len(sticks_gt))
            # Information on the gt angle and the classification success:
            # For 1 fiber, we keep in memory the spherical coordinate of the fiber
            if len(angle_gt) == 1:
                angle_information[0].append(
                    [success_rate[0][-1], angle_gt[0][0], angle_gt[0][1], angular_error[0][-1], over_estimated_fiber[0][-1],
                     under_estimated_fiber[0][-1], over_estimated_fiber_total[0][-1], under_estimated_fiber_total[0][-1]])
            # For 2 fibers, we keep in memory the angle beetween the two fibers
            elif len(angle_gt) == 2:
                a1, a2 = _check_directions(angle_gt)
                t = np.arccos(a1.dot(a2)) * 180 / np.pi
                angle_information[1].append([success_rate[1][-1], t, angular_error[1][-1], over_estimated_fiber[1][-1],
                                             under_estimated_fiber[1][-1], over_estimated_fiber_total[1][-1],
                                             under_estimated_fiber_total[1][-1]])
            # For 3 and 4 fibers, I don't have an idea yet
            else:
                angle_information[len(sticks_gt) - 1].append(
                    [success_rate[len(sticks_gt) - 1][-1], angular_error[len(sticks_gt) - 1][-1],
                     over_estimated_fiber[len(sticks_gt) - 1][-1], under_estimated_fiber[len(sticks_gt) - 1][-1],
                     over_estimated_fiber_total[len(sticks_gt) - 1][-1],
                     under_estimated_fiber_total[len(sticks_gt) - 1][-1]])
    theta = 0
    n_plus = 0
    n_moins = 0
    sr = 0
    N_t = 0
    for i in range(max_f):
        print("GT %i fibers:" % (i + 1))
        print('N')
        N = len(under_estimated_fiber_total[i])
        N_t += N
        print(N)
        print('angle')
        ae = np.mean(angular_error[i]) * 180 / np.pi
        theta += N * ae
        print(ae)
        print('classification')
        c = np.sum(np.array(success_rate[i]) == 1) / len(success_rate[i])
        sr += N * c
        print(c)
        print('over-estimation')
        oe = np.mean(over_estimated_fiber[i])
        n_plus += N * oe
        print(oe)
        print('under-estimation')
        ue = np.mean(under_estimated_fiber[i])
        n_moins += N * ue
        print(ue)
        print('over-estimation total')
        oet = np.mean(over_estimated_fiber_total[i])
        print(oet)
        print('under-estimation total')
        uet = np.mean(under_estimated_fiber_total[i])
        print(uet)
    print('-----------------TOTAL------------------')
    print('N')
    print(N_t)
    print('angle')
    print(theta / N_t)
    print('classification')
    print(sr / N_t)
    print('over-estimation')
    print(n_plus / N_t)
    print('under-estimation')
    print(n_moins / N_t)

    return angular_error, success_rate, over_estimated_fiber, under_estimated_fiber, angle_information
