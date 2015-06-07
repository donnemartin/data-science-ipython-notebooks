from __future__ import division
import numpy as np

__author__ = "Eric Chiang"
__email__  = "eric[at]yhathq.com"

"""

Measurements inspired by Philip Tetlock's "Expert Political Judgment"

Equations take from Yaniv, Yates, & Smith (1991):
  "Measures of Descrimination Skill in Probabilistic Judgement"

"""


def calibration(prob,outcome,n_bins=10):
    """Calibration measurement for a set of predictions.

    When predicting events at a given probability, how far is frequency
    of positive outcomes from that probability?
    NOTE: Lower scores are better

    prob: array_like, float
        Probability estimates for a set of events

    outcome: array_like, bool
        If event predicted occurred

    n_bins: int
        Number of judgement categories to prefrom calculation over.
        Prediction are binned based on probability, since "descrete" 
        probabilities aren't required. 

    """
    prob = np.array(prob)
    outcome = np.array(outcome)

    c = 0.0
    # Construct bins
    judgement_bins = np.arange(n_bins + 1) / n_bins
    # Which bin is each prediction in?
    bin_num = np.digitize(prob,judgement_bins)
    for j_bin in np.unique(bin_num):
        # Is event in bin
        in_bin = bin_num == j_bin
        # Predicted probability taken as average of preds in bin
        predicted_prob = np.mean(prob[in_bin])
        # How often did events in this bin actually happen?
        true_bin_prob = np.mean(outcome[in_bin])
        # Squared distance between predicted and true times num of obs
        c += np.sum(in_bin) * ((predicted_prob - true_bin_prob) ** 2)
    return c / len(prob)

def discrimination(prob,outcome,n_bins=10):
    """Discrimination measurement for a set of predictions.

    For each judgement category, how far from the base probability
    is the true frequency of that bin?
    NOTE: High scores are better

    prob: array_like, float
        Probability estimates for a set of events

    outcome: array_like, bool
        If event predicted occurred

    n_bins: int
        Number of judgement categories to prefrom calculation over.
        Prediction are binned based on probability, since "descrete" 
        probabilities aren't required. 

    """
    prob = np.array(prob)
    outcome = np.array(outcome)

    d = 0.0
    # Base frequency of outcomes
    base_prob = np.mean(outcome)
    # Construct bins
    judgement_bins = np.arange(n_bins + 1) / n_bins
    # Which bin is each prediction in?
    bin_num = np.digitize(prob,judgement_bins)
    for j_bin in np.unique(bin_num):
        in_bin = bin_num == j_bin
        true_bin_prob = np.mean(outcome[in_bin])
        # Squared distance between true and base times num of obs
        d += np.sum(in_bin) * ((true_bin_prob - base_prob) ** 2)
    return d / len(prob)
