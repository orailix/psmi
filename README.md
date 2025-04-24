# Numerical estimator of Pointwise Sliced Mutual Information (PSMI).

*Jérémie Dentan, École Polytechnique*

## Setup

Our library is published on PyPI!

```bash
pip install psmi
```

## Usage

We implement a class `PSMI` which should be used in a similar way to scikit-learn classes, with `fit`, `transform` and `fit_transform` methods.

We only implement PSMI between scalar feature and integer labels
belonging to a finite number of classes. We use algorithm 1 in [1] to
estimate PSMI. The only hyperparameter of this algorithm is the number
of estimator (i.e. the number of directions samples to estimate PSMI).

We propose two approaches to compute PSMI.

1. Manual. You simply pass argument `n_estimators` with the desired value.

2. Automatic. In that cas, you pass `n_estimators="auto"` and an
algorithm will be used to determine a suitable value for `n_estimators`.

### Example

```python
import numpy as np
from psmi import PSMI

# Generating data
n_samples, dim, n_labels = 100, 1024, 5
features = np.random.random((n_samples, dim))
labels = np.random.randint(n_labels, size=n_samples)

# Manual number of estimator
psmi_estimator = PSMI(n_estimators=500)
psmi_mean, psmi_std, psmi_full = psmi_estimator.fit_transform(features, labels)
print(f"psmi_mean: {psmi_mean.shape}")  # Should be (100,)
print(f"psmi_std: {psmi_std.shape}")  # Should be (100,)
print(f"psmi_full: {psmi_full.shape}")  # Should be (100,500)
print(f"Num of estimator: {psmi_estimator.n_estimators}")  # Should be 500

# Automatic number of estimator
psmi_estimator = PSMI()
psmi_mean, psmi_std, psmi_full = psmi_estimator.fit_transform(features, labels)
print(f"psmi_mean: {psmi_mean.shape}")  # Should be (100,)
print(f"psmi_std: {psmi_std.shape}")  # Should be (100,)
print(f"psmi_full: {psmi_full.shape}")  # Should be (100,<psmi_estimator.n_estimators>)
print(f"Num of estimator: {psmi_estimator.n_estimators}")

# You can separate the fit and transform
n_test = 5
features_test = np.random.random((n_test, dim))
labels_test = np.random.randint(n_labels, size=n_test)
psmi_estimator = PSMI()
psmi_estimator.fit(features, labels)
psmi_mean, psmi_std, psmi_full = psmi_estimator.transform(features_test, labels_test)
print(f"psmi_mean: {psmi_mean.shape}")  # Should be (5,)
print(f"psmi_std: {psmi_std.shape}")  # Should be (5,)
print(f"psmi_full: {psmi_full.shape}")  # Should be (5,<psmi_estimator.n_estimators>)
print(f"Num of estimator: {psmi_estimator.n_estimators}")
```

## Details on the `auto` mode

More specifically, we will iteratively add more and more estimators, in
blocks of `min_n_estimators`. We stop this process when the PSMI of
the elements that have the lowest PSMI minimally evolved between the current
step and the one with half as many estimators.

More specifically, if `lowest_psmi_quantile=0.05`, we consider the 5%
of elements with the lowest PSMI at current step. Then, we compare this
value to the PSMI of theses elements using only the first `int(n*milestone)`
blocks of estimators, where `n` is the current number of blocks that was
added. Then we compare the absolute value of the variation divided by the
PSMI at the current step. If it is below `max_variation_of_the_lowest`,
we stop. Else, we add another block of `min_n_estimators` estimators.

For example, the default values corresponds to blocks of 500 estimators.
We add blocks untill the 5% of elements with lowest PSMI have varied of
less than 5% between the current step and the one with half as many blocks.

[1] Shelvia Wongso et al. Pointwise Sliced Mutual Information for Neural
Network Explainability. IEEE International Symposium on Information
Theory (ISIT). 2023. DOI: 10.1109/ISIT54713.2023.10207010

## Contributing

You are welcome to submit pull requests! Please use pre-commit to correctly format your code:

```bash
pip install -r .github/dev-requirements.txt
pre-commit install
```

Please test your code:

```bash
pytest
```

## License and Copyright

Copyright 2024-present Laboratoire d'Informatique de Polytechnique. This project is licensed under the GNU Lesser General Public License v3.0. See the LICENSE file for details.

Please cite this work as follows:

```bibtex
@misc{dentan_predicting_2024,
	title = {Predicting and analysing memorization within fine-tuned Large Language Models},
	url = {https://arxiv.org/abs/2409.18858},
	author = {Dentan, Jérémie and Buscaldi, Davide and Shabou, Aymen and Vanier, Sonia},
	month = sep,
	year = {2024},
}
```

## Acknowldgements

This work received financial support from Crédit Agricole SA through the research chair ”Trustworthy and responsible AI” with École Polytechnique.
