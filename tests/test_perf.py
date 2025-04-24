# `psmi`

# Copyright 2024-present Laboratoire d'Informatique de Polytechnique.
# License LGPL-3.0

from time import time

import numpy as np
import pytest
import torch

from psmi import PSMI
from tests.data import get_test_data
from tests.groundtruth import get_ground_truth


def test_fit_with_200_estimators():

    # Loading
    features, labels = get_test_data()
    gt_mean, gt_std = get_ground_truth()

    # Init estimators
    est_0, est_1 = PSMI(n_estimators=500), PSMI(n_estimators=500)
    assert (not est_0.fitted) and (not est_1.fitted)

    # Fit and transform
    t_init = time()
    psmi_mean_0, psmi_std_0, psmi_0 = est_0.fit(features, labels).transform(
        features, labels
    )
    delta_t_0 = time() - t_init
    psmi_mean_1, psmi_std_1, psmi_1 = est_1.fit_transform(features, labels)
    delta_t_1 = time() - t_init - delta_t_0

    # Values
    assert np.mean(np.abs(psmi_mean_0 - gt_mean)) < 0.05
    assert np.mean(np.abs(psmi_mean_1 - gt_mean)) < 0.05
    assert np.mean(np.abs(psmi_std_0 - gt_std)) < 0.05
    assert np.mean(np.abs(psmi_std_1 - gt_std)) < 0.05

    # Dimensions
    assert (est_0.fitted) and (est_1.fitted)
    assert psmi_0.shape == psmi_1.shape == (10000, 500)
    assert est_0.thetas.shape == est_1.thetas.shape == (500, 512)
    assert len(est_0.frequency_per_label) == len(est_1.frequency_per_label) == 10
    assert len(est_0.mean_per_label) == len(est_1.mean_per_label) == 10
    assert len(est_0.std_per_label) == len(est_1.std_per_label) == 10
    assert est_0.mean_per_label[0].shape == est_1.mean_per_label[0].shape == (500,)
    assert est_0.std_per_label[0].shape == est_1.std_per_label[0].shape == (500,)

    # Runtime - we do not test it precisely because of the variability
    assert (delta_t_0 - delta_t_1) > 0


def test_auto_n_est():

    # Loading
    features, labels = get_test_data()
    gt_mean, gt_std = get_ground_truth()

    # Init estimators
    est_0, est_1 = PSMI(), PSMI()
    assert (not est_0.fitted) and (not est_1.fitted)

    # Fit and transform
    psmi_mean_0, psmi_std_0, psmi_0 = est_0.fit(features, labels).transform(
        features, labels
    )
    psmi_mean_1, psmi_std_1, psmi_1 = est_1.fit_transform(features, labels)

    # Values
    assert np.mean(np.abs(psmi_mean_0 - gt_mean)) < 0.01
    assert np.mean(np.abs(psmi_mean_1 - gt_mean)) < 0.01
    assert np.mean(np.abs(psmi_std_0 - gt_std)) < 0.01
    assert np.mean(np.abs(psmi_std_1 - gt_std)) < 0.01

    # Num of estimators
    assert est_0.n_estimators >= 3500 and est_0.n_estimators <= 6000
    assert est_1.n_estimators >= 3500 and est_1.n_estimators <= 6000

    # Elements with lowest PSMI
    milestone_0 = 500 * (est_0.n_estimators // 500 // 2)
    milestone_1 = 500 * (est_1.n_estimators // 500 // 2)
    idx_low_0 = psmi_mean_0.argsort()[: int(0.05 * 10000)]
    idx_low_1 = psmi_mean_1.argsort()[: int(0.05 * 10000)]
    mean_half_0 = np.mean(psmi_0[:, :milestone_0], axis=1)
    mean_half_1 = np.mean(psmi_1[:, :milestone_1], axis=1)
    variations_0 = np.abs((mean_half_0 - psmi_mean_0) / psmi_mean_0)
    variations_1 = np.abs((mean_half_1 - psmi_mean_1) / psmi_mean_1)
    assert np.mean(variations_0[idx_low_0]) <= 0.05
    assert np.mean(variations_1[idx_low_1]) <= 0.05

    # Dimensions
    assert (est_0.fitted) and (est_1.fitted)
    assert est_0.thetas.shape == (est_0.n_estimators, 512)
    assert est_1.thetas.shape == (est_1.n_estimators, 512)
    assert len(est_0.frequency_per_label) == len(est_1.frequency_per_label) == 10
    assert len(est_0.mean_per_label) == len(est_1.mean_per_label) == 10
    assert len(est_0.std_per_label) == len(est_1.std_per_label) == 10
    assert est_0.mean_per_label[0].shape == (est_0.n_estimators,)
    assert est_1.mean_per_label[0].shape == (est_1.n_estimators,)
    assert est_0.std_per_label[0].shape == (est_0.n_estimators,)
    assert est_1.std_per_label[0].shape == (est_1.n_estimators,)

    # Runtime - we don't test it because depending on the n_est found it can be inverted


def test_milabelled_samples():

    # Loading
    features, labels = get_test_data()
    gt_mean, gt_std = get_ground_truth()

    # Mislabel
    labels[-100:] = torch.tensor(np.random.randint(0, 10, size=100))

    # PSMI
    psmi_mean, _, _ = PSMI().fit_transform(features, labels)

    # Values
    assert np.mean(np.abs(psmi_mean[:-100] - gt_mean[:-100])) < 0.01
    assert np.mean(np.abs(psmi_mean[-100:] - gt_mean[-100:])) > 0.12
