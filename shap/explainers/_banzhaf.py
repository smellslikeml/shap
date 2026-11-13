from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .. import links
from .._cutils import compute_grey_code_row_values, compute_grey_code_row_values_st
from ..utils import MaskedModel
from ..utils._general import banzhaf_coefficients
from ._exact import ExactExplainer


class BanzhafExplainer(ExactExplainer):
    """Computes Banzhaf values via an optimized exact enumeration.

    This explainer mirrors the non-clustering path of ``shap.explainers.ExactExplainer``: all
    2**n coalitions are enumerated in gray code order and the resulting marginals are combined
    with the same compiled kernel. The only difference is the coalition weighting. The Shapley
    value weights a marginal contribution over a coalition of size ``s`` by
    ``1 / (n * comb(n - 1, s))``, while the Banzhaf value (power index) weights every marginal
    contribution uniformly by ``1 / 2**(n - 1)``. The Banzhaf value of a feature is therefore its
    average marginal contribution over all the coalitions it could join.

    Note that unlike Shapley values, Banzhaf values are not *efficient*: they do not generally sum
    to the difference between the model output and the expected model output, so
    ``base_values + values.sum()`` should not be expected to reconstruct the model prediction.

    Like the Exact explainer this works well for standard maskers with less than ~15 features that
    vary from the background per sample. Clustering maskers are not supported since the Banzhaf
    value is defined over flat (unstructured) coalition games.

    Examples
    --------
    The interface is shared with `shap.explainers.ExactExplainer
    <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/explainers/Exact.html>`_, e.g.::

        explainer = shap.BanzhafExplainer(model.predict, background_data)
        shap_values = explainer(data_to_explain)
    """

    def __init__(
        self,
        model: Any,
        masker: Any,
        link: Any = links.identity,
        linearize_link: bool = True,
        feature_names: Any = None,
    ) -> None:
        """Build an explainers.Banzhaf object for the given model using the given masker object.

        Parameters
        ----------
        model : function
            A callable python object that executes the model given a set of input data samples.

        masker : function or numpy.array or pandas.DataFrame
            A callable python object used to "mask" out hidden features of the form `masker(mask, *fargs)`.
            It takes a single a binary mask and an input sample and returns a matrix of masked samples. These
            masked samples are evaluated using the model function and the outputs are then averaged.
            As a shortcut for the standard masking used by SHAP you can pass a background data matrix
            instead of a function.

        link : function
            The link function used to map between the output units of the model and the SHAP value units. By
            default it is shap.links.identity, but shap.links.logit can be useful so that expectations are
            computed in probability units while explanations remain in the (more naturally additive) log-odds
            units. For more details on how link functions work see any overview of link functions for generalized
            linear models.

        linearize_link : bool
            If we use a non-linear link function to take expectations then models that are additive with respect to that
            link function for a single background sample will no longer be additive when using a background masker with
            many samples. This for example means that a linear logistic regression model would have interaction effects
            that arise from the non-linear changes in expectation averaging. To retain the additively of the model with
            still respecting the link function we linearize the link function by default.

        """
        if getattr(masker, "clustering", None) is not None:
            raise ValueError(
                "The Banzhaf explainer is defined over flat coalition games and does not support clustering maskers! "
                "Please use a plain (e.g. Independent) masker, or use shap.explainers.ExactExplainer instead."
            )
        super().__init__(model, masker, link=link, linearize_link=linearize_link, feature_names=feature_names)

    def explain_row(  # type: ignore[override]
        self,
        *row_args: Any,
        max_evals: int | Literal["auto"],
        main_effects: bool,
        error_bounds: bool,
        batch_size: int | Literal["auto"],
        outputs: Any,
        interactions: bool | int,
        silent: bool,
    ) -> dict[str, Any]:
        """Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes)."""
        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, self.linearize_link, *row_args)

        # see which elements we actually need to perturb
        inds = fm.varying_inputs()
        # varying_inputs() returns np.intp, which is int32 on Windows; copy=False
        # makes this free on platforms where it is already int64
        inds = inds.astype(np.int64, copy=False)
        if len(inds) == 0:
            # if nothing varies then we can just return the expected value as the output and be done with it
            outputs = fm(np.array([MaskedModel.delta_mask_noop_value]), zero_index=0, batch_size=batch_size)
            return {
                "values": np.zeros(row_args[0].shape),
                "expected_values": outputs[0],
                "mask_shapes": fm.mask_shapes,
                "main_effects": None,
                "clustering": None,
            }

        # make sure we have enough evals
        if max_evals is not None and max_evals != "auto" and max_evals < 2 ** len(inds):
            raise ValueError(
                f"It takes {2 ** len(inds)} masked evaluations to run the Banzhaf explainer on this instance, "
                f"but max_evals={max_evals}!"
            )

        # generate the masks in gray code order (so that we change the inputs as little
        # as possible while we iterate to minimize the need to re-eval when the inputs
        # don't vary from the background)
        delta_indexes = self._cached_gray_codes(len(inds))

        # map to a larger mask that includes the invariant entries
        extended_delta_indexes = np.zeros(2 ** len(inds), dtype=np.int64)
        for i in range(2 ** len(inds)):
            if delta_indexes[i] == MaskedModel.delta_mask_noop_value:
                extended_delta_indexes[i] = delta_indexes[i]
            else:
                extended_delta_indexes[i] = inds[delta_indexes[i]]

        # run the model
        outputs = fm(extended_delta_indexes, zero_index=0, batch_size=batch_size)

        # Banzhaf values
        # Care: Need to distinguish between `True` and `1`
        if interactions is False or (interactions == 1 and interactions is not True):
            # loop over all the outputs to update the rows
            coeff = banzhaf_coefficients(len(inds))
            row_values = np.zeros((len(fm),) + outputs.shape[1:], dtype=np.float64)
            mask = np.zeros(len(fm), dtype=bool)

            compute_grey_code_row_values(
                row_values, mask, inds, outputs, coeff, extended_delta_indexes, MaskedModel.delta_mask_noop_value
            )

        # pairwise interaction values using the same uniform coalition weighting
        elif interactions is True or interactions == 2:
            # loop over all the outputs to update the rows
            coeff = banzhaf_coefficients(len(inds))
            row_values = np.zeros((len(fm), len(fm)) + outputs.shape[1:], dtype=np.float64)
            mask = np.zeros(len(fm), dtype=bool)
            compute_grey_code_row_values_st(
                row_values, mask, inds, outputs, coeff, extended_delta_indexes, MaskedModel.delta_mask_noop_value
            )

        elif interactions > 2:
            raise NotImplementedError(
                "Currently the Banzhaf explainer does not support interactions higher than order 2!"
            )

        # compute the main effects if we need to
        main_effect_values = None
        if main_effects or interactions is True or interactions == 2:
            main_effect_values = fm.main_effects(inds)
            if interactions is True or interactions == 2:
                for i in range(len(fm)):
                    row_values[i, i] = main_effect_values[i]

        return {
            "values": row_values,
            "expected_values": outputs[0],
            "mask_shapes": fm.mask_shapes,
            "main_effects": main_effect_values if main_effects else None,
            "clustering": None,
        }

    def __str__(self) -> str:
        return "shap.explainers.BanzhafExplainer()"
