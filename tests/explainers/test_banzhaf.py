"""This file contains tests for the Banzhaf explainer."""

import numpy as np
from conftest import compare_numpy_outputs_against_baseline

import shap
from shap.utils._general import banzhaf_coefficients, shapley_coefficients


def _weighted_voting_model(weights, quota):
    """Return a model whose output is 1 when the weights of the present features meet the quota, else 0.

    With an all-zero background row and an all-one row to explain, the masked evaluation rows encode
    coalitions directly: feature j is present (1) exactly when it belongs to the coalition, so the
    model output is the value v(S) of the weighted voting game.
    """

    def model(X):
        return (X @ np.asarray(weights) >= quota).astype(np.float64)

    return model


def test_banzhaf_coefficients():
    """Every coalition size gets the same uniform weight, unlike the size-dependent Shapley weights."""
    np.testing.assert_allclose(banzhaf_coefficients(3), [0.25, 0.25, 0.25])
    np.testing.assert_allclose(banzhaf_coefficients(4), [0.125] * 4)
    assert not np.allclose(banzhaf_coefficients(4), shapley_coefficients(4))


def test_registration():
    """The explainer is registered like the other explainers."""
    assert shap.BanzhafExplainer is shap.explainers.BanzhafExplainer
    assert shap.explainers.Banzhaf is shap.BanzhafExplainer
    assert "BanzhafExplainer" in shap.__all__


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_weighted_voting_game():
    """Check against the hand-computed Banzhaf values of a known weighted voting game.

    The game has players with weights [3, 2, 1] and a bill passes when the present weights sum to
    at least the quota 4. Each player's Banzhaf value is its average marginal contribution over
    the 2**(3 - 1) = 4 coalitions it could join, giving [3/4, 1/4, 1/4].

    Note the values do NOT sum to v(N) - v({}) = 1: unlike Shapley values, Banzhaf values are not
    efficient, so no additivity (efficiency) assertion is made for this explainer.
    """
    explainer = shap.BanzhafExplainer(_weighted_voting_model([3.0, 2.0, 1.0], 4), np.zeros((1, 3)))
    shap_values = explainer(np.ones((1, 3)))

    np.testing.assert_allclose(shap_values.values[0], [0.75, 0.25, 0.25], atol=1e-10)
    # not efficient: the values sum to 1.25, not to v(N) - v({}) = 1
    np.testing.assert_allclose(shap_values.values.sum(), 1.25, atol=1e-10)
    return shap_values


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_dummy_player():
    """A player that never changes the game outcome gets a Banzhaf value of zero."""
    # feature 3 has weight 0: it varies from the background but is never pivotal,
    # and the real players keep the values they had in the 3 player game
    explainer = shap.BanzhafExplainer(_weighted_voting_model([3.0, 2.0, 1.0, 0.0], 4), np.zeros((1, 4)))
    shap_values = explainer(np.ones((1, 4)))

    np.testing.assert_allclose(shap_values.values[0, 3], 0.0, atol=1e-10)
    np.testing.assert_allclose(shap_values.values[0, :3], [0.75, 0.25, 0.25], atol=1e-10)
    return shap_values


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_symmetry():
    """Interchangeable players receive equal Banzhaf values."""
    # in this majority game every player is interchangeable, so all values are equal
    # (and each matches the closed-form average marginal contribution of 2/4)
    explainer = shap.BanzhafExplainer(_weighted_voting_model([1.0, 1.0, 1.0], 2), np.zeros((1, 3)))
    shap_values = explainer(np.ones((1, 3)))

    values = shap_values.values[0]
    np.testing.assert_allclose(values, values[0], atol=1e-10)
    np.testing.assert_allclose(values, 0.5, atol=1e-10)
    return shap_values


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_linearity():
    """The Banzhaf value is linear in the game it explains."""
    voting = _weighted_voting_model([3.0, 2.0, 1.0], 4)

    def additive(X):
        return X @ np.array([1.0, 2.0, 3.0])

    def combined(X):
        return voting(X) + 2 * additive(X)

    background = np.zeros((1, 3))
    row = np.ones((1, 3))
    voting_values = shap.BanzhafExplainer(voting, background)(row).values
    additive_values = shap.BanzhafExplainer(additive, background)(row).values
    combined_values = shap.BanzhafExplainer(combined, background)(row).values

    np.testing.assert_allclose(combined_values, voting_values + 2 * additive_values, atol=1e-10)
    # for an additive game every marginal contribution is the feature's own coefficient
    np.testing.assert_allclose(additive_values, [1.0, 2.0, 3.0], atol=1e-10)
    return combined_values
