# -*- coding: utf-8 -*-
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compare_layers_per_ascan(data_gt, data_pred, layers=["BM", "RPE"]):
    """Compare layer annotations for two OCT volumes."""
    drusen_map = data_gt.drusen_projection > 0
    result = {}
    for layer in layers:
        dist = data_gt.layers[layer] - data_pred.layers[layer]
        for i, bscan in enumerate(data_gt):
            i += 1
            for j in range(dist.shape[1]):
                if not np.isnan(dist[-i, j]):
                    result[(f"{layer}_Distance", bscan.name, j)] = [dist[-i, j]]
                    if drusen_map[-i, j]:
                        result[(f"Drusen Load", bscan.name, j)] = "Drusen AScan"
                    else:
                        result[(f"Drusen Load", bscan.name, j)] = "No Drusen AScan"
    return result


def compare_drusen_per_bscan(data_gt, data_pred):
    """Compare drusen annotations for two OCT volumes."""
    result = {}
    for bscan_gt, bscan_pred in zip(data_gt, data_pred):
        drusen_gt = bscan_gt.drusen
        drusen_pred = bscan_pred.drusen

        union = np.sum(np.logical_or(drusen_gt, drusen_pred))
        intersection = np.sum(np.logical_and(drusen_gt, drusen_pred))
        # Square for the denominator
        denominator = np.sum(drusen_gt ** 2) + np.sum(drusen_pred ** 2)
        if denominator != 0:
            dice = (2.0 * intersection + 1e-5) / (denominator)
        else:
            dice = 1

        difference = drusen_pred.astype(int) - drusen_gt.astype(int)
        false_positives = np.sum(difference > 0)
        false_negatives = np.sum(difference < 0)
        precision = (
            intersection / (intersection + false_positives)
            if false_positives != 0
            else 1
        )
        recall = (
            intersection / (intersection + false_negatives)
            if false_negatives != 0
            else 1
        )

        # If the union is 0, neither ground_truth nor prediction contain anything -> IoU becomes 1
        iou = intersection / union if union != 0 else 1

        # compute absolute area difference
        # abs_area_diff = np.abs(np.sum(drusen_gt) - np.sum(drusen_pred))
        # Normalize by the number of AScans with drusen
        # drusen_ascans = np.sum(np.sum(union, axis=0) > 0) if np.any(union) else 1.0
        # adad = abs_area_diff / drusen_ascans * 100
        # result[("ADAD", bscan_gt.name)] = [adad]

        result[("IoU", bscan_gt.name)] = [iou]
        result[("Dice", bscan_gt.name)] = [dice]
        result[("Precision", bscan_gt.name)] = [precision]
        result[("Recall", bscan_gt.name)] = [recall]
        result[("GT Drusen Volume", bscan_gt.name)] = [np.sum(drusen_gt)]
        result[("PR Drusen Volume", bscan_gt.name)] = [np.sum(drusen_pred)]

        drusen_load = np.sum(drusen_gt)
        if drusen_load == 0:
            result[("Drusen Load", bscan_gt.name)] = ["No Drusen"]
        elif drusen_load <= 150:
            result[("Drusen Load", bscan_gt.name)] = ["Drusen<150"]
        elif drusen_load <= 1000:
            result[("Drusen Load", bscan_gt.name)] = ["Drusen<1000"]
        else:
            result[("Drusen Load", bscan_gt.name)] = ["Drusen>1000"]

    return result


def compare_annotations(data_gt, data_pred, layers=["BM", "RPE"]):
    """Compare layer and drusen annotations for two OCT volumes.

    Return two pandas DataFrames. One for BScan wise Area measures and
    one for an AScan wise layer height comparison.
    """

    layer_results = compare_layers_per_ascan(data_gt, data_pred, layers=["BM", "RPE"])
    area_results = compare_drusen_per_bscan(data_gt, data_pred)

    area_df = pd.DataFrame.from_dict(area_results).T
    area_df.index.names = ["Metric", "BScan"]
    area_df = pd.melt(area_df.unstack().T, id_vars=["Drusen Load"], ignore_index=False)
    area_df = area_df.reset_index().set_index(["BScan"]).drop("level_0", axis=1)

    layer_df = pd.DataFrame.from_dict(layer_results).T
    layer_df.index.names = ["Metric", "BScan", "AScan"]
    layer_df = pd.melt(
        layer_df.unstack([-1, -2]).T, id_vars=["Drusen Load"], ignore_index=False
    )
    layer_df = (
        layer_df.reset_index().set_index(["BScan", "AScan"]).drop("level_0", axis=1)
    )

    return area_df, layer_df
