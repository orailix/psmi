# `psmi`

# Copyright 2024-present Laboratoire d'Informatique de Polytechnique.
# License LGPL-3.0

import os

import numpy as np
import torch

from tests.data import get_test_data

data_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data"))
ground_truth_mean_path = os.path.join(data_root, "ground_truth_mean.pt")
ground_truth_std_path = os.path.join(data_root, "ground_truth_std.pt")


def compute():

    from psmi import PSMI

    features, labels = get_test_data()
    psmi_mean, psmi_std, _ = PSMI(n_estimators=20000).fit_transform(features, labels)

    torch.save(torch.tensor(psmi_mean), ground_truth_mean_path)
    torch.save(torch.tensor(psmi_std), ground_truth_std_path)


def get_ground_truth():

    if not os.path.isfile(ground_truth_mean_path) or not os.path.isfile(
        ground_truth_std_path
    ):
        compute()

    return (
        np.array(
            torch.load(ground_truth_mean_path, weights_only=True),
            copy=True,
            dtype=float,
        ),
        np.array(
            torch.load(ground_truth_std_path, weights_only=True),
            copy=True,
            dtype=float,
        ),
    )
