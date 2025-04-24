# `psmi`

# Copyright 2024-present Laboratoire d'Informatique de Polytechnique.
# License LGPL-3.0

import typing as t

import numpy as np


class PSMI:
    """Numerical estimator of Pointwise Sliced Mutual Information (PSMI).

    We only implement PSMI between scalar feature and integer labels
    belonging to a finite number of classes. We use algorithm 1 in [1] to
    estimate PSMI. The only hyperparameter of this algorithm is the number
    of estimator (i.e. the number of directions samples to estimate PSMI).

    We propose two approaches to compute PSMI.

    1. Manual. You simply pass argument `n_estimators` with the desired value.

    2. Automatic. In that cas, you pass `n_estimators="auto"` and an
    algorithm will be used to determine a suitable value for `n_estimators`.

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

    Parameters
    ----------
    n_estimators :
        Either the number of estimators, or "auto"
    min_n_estimators :
        The size of the blocks of estimators that will be added in "auto" mode
    lowest_psmi_quantile :
        The quantile to select elements with lowest PSMI in "auto" mode
    max_variation_of_the_lowest :
        The maximum variation of PSMI used in "auto" mode
    milestone :
        The milestone used in "auto" mode
    """

    def __init__(
        self,
        n_estimators: t.Union[str, int] = "auto",
        min_n_estimators: int = 500,
        lowest_psmi_quantile: float = 0.05,
        max_variation_of_the_lowest: float = 0.05,
        milestone: float = 0.5,
    ) -> None:

        # Parsing n_nestimators
        if n_estimators == "auto":
            self.n_estimators: t.Optional[int] = None
        else:
            self.n_estimators: t.Optional[int] = int(n_estimators)

        # Default hyperparameters
        self.min_n_estimators: int = int(min_n_estimators)
        self.lowest_psmi_quantile: float = float(lowest_psmi_quantile)
        self.lowest_psmi_quantile: float = float(lowest_psmi_quantile)
        self.max_variation_of_the_lowest: float = float(max_variation_of_the_lowest)
        self.milestone: float = float(milestone)

        # Init attributes to be fitted
        self._fitted: bool = False
        self._thetas: t.Optional[np.ndarray] = None
        self._frequency_per_label: t.Optional[t.Dict[int, float]] = None
        self._mean_per_label: t.Optional[t.Dict[int, np.ndarray]] = None
        self._std_per_label: t.Optional[t.Dict[int, np.ndarray]] = None

        # Attributes for fit-transform
        self._saved_projected_features: t.Optional[np.ndarray] = None

    @property
    def fitted(self) -> bool:
        """Whether or not the instance have already been fitted."""
        return self._fitted

    @property
    def thetas(self) -> t.Optional[np.ndarray]:
        """The array of directions, of shape (n_estimators, dim).

        Is None when the instance is not fitted."""
        return self._thetas

    @property
    def frequency_per_label(self) -> t.Optional[t.Dict[int, float]]:
        """A dictionnary containing the frequency of each label.

        Is None when the instance is not fitted."""
        return self._frequency_per_label

    @property
    def mean_per_label(self) -> t.Optional[t.Dict[int, np.ndarray]]:
        """A dictionnary containing the estimated mean along every
        direction theta per label.

        `self.mean_per_label[0]` is an array of shape (n_estimators,)
        containing the mean of the projected features having this label
        along each direction theta.

        Is None when the instance is not fitted.
        """
        return self._mean_per_label

    @property
    def std_per_label(self) -> t.Optional[t.Dict[int, np.ndarray]]:
        """A dictionnary containing the estimated std along every
        direction theta per label.

        `self.mean_per_label[0]` is an array of shape (n_estimators,)
        containing the std of the projected features having this label
        along each direction theta.

        Is None when the instance is not fitted.
        """
        return self._std_per_label

    @property
    def saved_projected_features(self) -> t.Optional[np.ndarray]:
        """An array containing the saved projected features.

        This is used with using `self.fit_transform` method, to avoid
        re-computing the projected features in the `fit` and `transform`
        parts.

        Is None when the instance is not fitted, or when the model is fitted
        without explicitely asking for `save_projected_features=True`."""
        return self._saved_projected_features

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_projected_features: bool = False,
    ) -> "PSMI":
        """Fits the PSMI numerical estimator.

        This steps consists of estimating the mean and standard deviation
        along all direction `self.thetas` grouped by possible labels.

        The additional parameter `save_projected_features` serves for
        saving the projected features to avoid re-computing them when
        using `self.fit_transform` method.

        Parameters
        ----------
        features :
            An array of shape (n_samples, dim) containing the features
        labels :
            An array of shape (n_samples,) containing the labels
        save_projected_features :
            A boolean indicating if the projected features must be saved
            or not.

        Returns
        -------
        self :
            The instance of PSMI

        """

        # ================ INITIALIZATION ================

        # Already fitter?
        if self._fitted:
            raise RuntimeError(
                "You are trying to fit a PSMI estimator which is already fitted."
            )

        # Need auto num of estimators?
        if self.n_estimators is None:
            return self._fit_auto_n_estimators(
                features,
                labels,
                save_projected_features,
            )

        # Convert to numpy
        if not isinstance(features, np.ndarray):
            features = np.array(features, copy=True, dtype=float)

        if not isinstance(labels, np.ndarray):
            labels = np.array(labels, copy=True, dtype=float)

        # Dimensions
        if len(features.shape) != 2:
            raise ValueError(
                f"Incompatible shape: features is {features.shape} but it should be (n_samples, dim)"
            )
        n_samples, dim = features.shape

        if len(labels.shape) != 1 or labels.shape[0] != n_samples:
            raise ValueError(
                f"Incompatible shapes: features is {features.shape} and labels is {labels.shape}, "
                "whereas we expected (n_samples, dim) for features and (n_samples,) for labels."
            )

        # ================ THETAS AND PROJECTED FEATURES ================

        # Sampling theta uniformly on the hypersphere
        # Theta will have size (n_estimators, dim), each row uniformly sampled on the dim-hypersphere
        # Reference: Marsaglia, G. (1972). “Choosing a Point from the Surface of a Sphere”...
        # ... Annals of Mathematical Statistics. 43 (2): 645-646.
        self._thetas = np.random.normal(0, 1, self.n_estimators * dim).reshape(
            self.n_estimators, dim
        )
        self._thetas = self._thetas / np.linalg.norm(self._thetas, axis=1)[:, None]

        # Projected features
        # dim is projected in a 1-dimensional space, so the feature tensor has size (n_samples, n_estimators)
        projected_features = features @ self._thetas.T

        # Index per possible categorical label
        unique_labels = np.unique(labels)
        label_to_sample_array = {
            label: np.where(labels == label)[0] for label in unique_labels
        }

        # Label frequencies
        self._frequency_per_label = {
            label: len(sample_array) / n_samples
            for label, sample_array in label_to_sample_array.items()
        }

        # Checking that each label has at least two samples
        for label, sample_array in label_to_sample_array.items():
            if len(sample_array) < 2:
                raise RuntimeError(
                    f"We cannot compute PSMI: the following label has less than 2 samples: {label}"
                )

        # ================ ESTIMATING MEAN AND STD ================

        # Filling the estimated mean and std
        self._mean_per_label = dict()
        self._std_per_label = dict()
        for possible_label, sample_array in label_to_sample_array.items():
            sample_features = projected_features[sample_array, :]
            self._mean_per_label[possible_label] = sample_features.mean(axis=0)
            self._std_per_label[possible_label] = sample_features.std(axis=0)

        # Save projected features?
        if save_projected_features:
            self._saved_projected_features = projected_features

        # ================ OUTPUT ================

        self._fitted = True
        return self

    def _fit_auto_n_estimators(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_projected_features: bool = False,
    ) -> "PSMI":
        """Fits the PSMI numerical estimator.

        Here, we have to estimate an adequate number of estimators.
        See the docstring of the class for more details on the algorithm.

        This steps consists of estimating the mean and standard deviation
        along all direction `self.thetas` grouped by possible labels.

        The additional parameter `save_projected_features` serves for
        saving the projected features to avoid re-computing them when
        using `self.fit_transform` method.

        Parameters
        ----------
        features :
            An array of shape (n_samples, dim) containing the features
        labels :
            An array of shape (n_samples,) containing the labels
        save_projected_features :
            A boolean indicating if the projected features must be saved
            or not.

        Returns
        -------
        self:
            The instance of PSMI

        """

        psmi_mean_per_block = None

        while True:

            # ================ FIT TRANSFORM ON A CHILD ================

            # Fit-transform on an instance using the minimum number of estimators
            # We don't use fit_transform to avoir the saved projected features to be deleted
            child = PSMI(n_estimators=self.min_n_estimators)
            child.fit(features, labels, save_projected_features=True)
            block_mean_psmi, _, _ = child.transform(
                features, labels, use_projected_features=True
            )

            # ================ SAVING ATTRS OF THE CHILD ================

            # PSMI
            if psmi_mean_per_block is None:
                psmi_mean_per_block = block_mean_psmi.reshape(-1, 1)
            else:
                psmi_mean_per_block = np.column_stack(
                    (psmi_mean_per_block, block_mean_psmi)
                )

            # Theta
            if self._thetas is None:
                self._thetas = child.thetas
            else:
                self._thetas = np.concatenate((self._thetas, child.thetas), axis=0)

            # Frequency - no need to update at each time
            if self._frequency_per_label is None:
                self._frequency_per_label = child.frequency_per_label

            # Mean per label
            if self._mean_per_label is None:
                self._mean_per_label = child.mean_per_label
            else:
                self._mean_per_label = {
                    label: np.concatenate(
                        (self._mean_per_label[label], child.mean_per_label[label])
                    )
                    for label in self._mean_per_label.keys()
                }

            # Std per label
            if self._std_per_label is None:
                self._std_per_label = child.std_per_label
            else:
                self._std_per_label = {
                    label: np.concatenate(
                        (self._std_per_label[label], child.std_per_label[label])
                    )
                    for label in self._std_per_label.keys()
                }

            # Saved projected features
            if save_projected_features:
                if self._saved_projected_features is None:
                    self._saved_projected_features = child.saved_projected_features
                else:
                    self._saved_projected_features = np.column_stack(
                        (self._saved_projected_features, child.saved_projected_features)
                    )

            # ================ STOPPING CRITERION ================

            # Init
            num_samples, num_batch = psmi_mean_per_block.shape
            num_lowest_psmi = int(self.lowest_psmi_quantile * num_samples)
            num_batch_milestone = int(self.milestone * num_batch)
            if num_batch_milestone == 0:
                continue

            # Full PSMI and variation from the milestone
            full_mean = psmi_mean_per_block.mean(axis=1)
            variation_to_full = np.abs(
                (psmi_mean_per_block[:, :num_batch_milestone].mean(axis=1) - full_mean)
                / full_mean
            )

            # Mean variation of the lowest
            idx_lowest_full_mean = full_mean.argsort()[:num_lowest_psmi]
            mean_variation_lowest = variation_to_full[idx_lowest_full_mean].mean()

            # Stopping criterion
            if mean_variation_lowest <= self.max_variation_of_the_lowest:
                break

        # ================ OUTPUT ================

        self.n_estimators = num_batch * self.min_n_estimators
        self._fitted = True
        return self

    def transform(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        use_projected_features: bool = False,
    ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes PSMI numerical estimator.

        The instance must have been already fitted.

        Parameters
        ----------
        features :
            An array of shape (n_samples, dim) containing the features
        labels :
            An array of shape (n_samples,) containing the labels
        use_projected_features :
            A boolean indicating if we must use the saved projected
            features or not. This is only the case when transforming on
            the same features that were used for fitting.

        Returns
        -------
        psmi_mean :
            An array of shape (n_samples,) containing, for each element
            of the dataset, the mean PSMI with respect to the choice of
            estimator (i.e. direction theta used for projection).
        psmi_std :
            An array of shape (n_samples,) containing, for each element
            of the dataset, the std PSMI with respect to the choice of
            estimator (i.e. direction theta used for projection).
        psmi :
            The full PSMI array, of shape (n_samples, n_estimators),
            containing the PSMI for every sample in every direction.

        """

        # ================ INIT ================

        # Already fitter?
        if not self._fitted:
            raise RuntimeError(
                "You are trying to transform from a PSMI estimator which is not fitted."
            )

        # Convert to numpy
        if not isinstance(features, np.ndarray):
            features = np.array(features, copy=True, dtype=float)

        if not isinstance(labels, np.ndarray):
            labels = np.array(labels, copy=True, dtype=float)

        # Dimensions
        if len(features.shape) != 2:
            raise ValueError(
                f"Incompatible shape: features is {features.shape} but it should be (n_samples, dim)"
            )
        n_samples, dim = features.shape

        if len(labels.shape) != 1 or labels.shape[0] != n_samples:
            raise ValueError(
                f"Incompatible shapes: features is {features.shape} and labels is {labels.shape}, "
                "whereas we expected (n_samples, dim) for features and (n_samples,) for labels."
            )

        # ================ PROJECTED FEATURES ================

        # Projected features
        # dim is projected in a 1-dimensional space, so the feature tensor has size (n_samples, n_estimator)
        if use_projected_features:
            projected_features = self._saved_projected_features
        else:
            projected_features = features @ self._thetas.T

        # Index per possible categorical label
        unique_labels = np.unique(labels)
        label_to_sample_array = {
            label: np.where(labels == label)[0] for label in unique_labels
        }

        # ================ COMPUTING PSMI ================

        # Pointise Mutual Information
        psmi_numerator = np.zeros_like(projected_features)
        psmi_denominator = np.zeros_like(projected_features)
        for label, sample_array in label_to_sample_array.items():

            # Numerator : we estimage p(projected_features | labels)
            psmi_numerator[sample_array, :] = norm_pdf(
                self._mean_per_label[label],
                self._std_per_label[label],
                projected_features[sample_array, :],
            )

            # Denominator: we use the formula of total probabilities
            # p(projected_features) = sum(p(labels)*p(projected_features|labels))
            label_freq = self._frequency_per_label[label]
            psmi_denominator += label_freq * norm_pdf(
                self._mean_per_label[label],
                self._std_per_label[label],
                projected_features,
            )

        # PMI array: shape n_samples, n_estimator
        # At index (i, j), if the j-th projected features of the i-th sample is x_ij
        # we have pmi[i,j] = log(p(x_ij|y_i) | p(x_ij))
        psmi = np.log(psmi_numerator / psmi_denominator)

        # ================ OUTPUT ================

        return np.mean(psmi, axis=1), np.std(psmi, axis=1), psmi

    def fit_transform(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fits the estimator and transforms it to compute PSMI.

        It's faster than using `self.fit` and `self.transform`
        sequentially, because we re-use the projected features
        without recomputing them.

        Parameters
        ----------
        features :
            An array of shape (n_samples, dim) containing the features
        labels :
            An array of shape (n_samples,) containing the labels

        Returns
        -------
        psmi_mean :
            An array of shape (n_samples,) containing, for each element
            of the dataset, the mean PSMI with respect to the choice of
            estimator (i.e. direction theta used for projection).
        psmi_std :
            An array of shape (n_samples,) containing, for each element
            of the dataset, the std PSMI with respect to the choice of
            estimator (i.e. direction theta used for projection).
        psmi :
            The full PSMI array, of shape (n_samples, n_estimators),
            containing the PSMI for every sample in every direction.

        """

        # Fitting and saving projected features
        self.fit(features, labels, save_projected_features=True)

        # Using saved features for transform
        output = self.transform(features, labels, use_projected_features=True)

        # Deleting projected features
        self._saved_projected_features = None

        # Output
        return output


def norm_pdf(mean: float, std: float, inputs: np.ndarray) -> np.ndarray:
    """Computes the value of the normal distribution for an array of inputs.

    Parameters
    ----------
    mean :
        The mean parameter of the normal distribution.
    std :
        The standard deviation parameter of the normal distribution.
    inputs :
        An array of shape (n_inputs,) containing the input values.

    Returns
    -------
    outputs :
        An array of shape (n_inputs,) containing the output values.

    """
    return (1 / std / np.sqrt(2 * np.pi)) * np.exp(
        -1 * (inputs - mean) ** 2 / 2 / std / std
    )
