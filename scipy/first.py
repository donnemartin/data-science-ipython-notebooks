"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function

import math
import numpy as np

import nsfg
import thinkstats2
import thinkplot


def MakeFrames():
    """Reads pregnancy data and partitions first babies and others.

    returns: DataFrames (all live births, first babies, others)
    """
    preg = nsfg.ReadFemPreg()

    live = preg[preg.outcome == 1]
    firsts = live[live.birthord == 1]
    others = live[live.birthord != 1]

    assert len(live) == 9148
    assert len(firsts) == 4413
    assert len(others) == 4735

    return live, firsts, others


def Summarize(live, firsts, others):
    """Print various summary statistics."""

    mean = live.prglngth.mean()
    var = live.prglngth.var()
    std = live.prglngth.std()

    print('Live mean', mean)
    print('Live variance', var)
    print('Live std', std)

    mean1 = firsts.prglngth.mean()
    mean2 = others.prglngth.mean()

    var1 = firsts.prglngth.var()
    var2 = others.prglngth.var()

    print('Mean')
    print('First babies', mean1)
    print('Others', mean2)

    print('Variance')
    print('First babies', var1)
    print('Others', var2)

    print('Difference in weeks', mean1 - mean2)
    print('Difference in hours', (mean1 - mean2) * 7 * 24)

    print('Difference relative to 39 weeks', (mean1 - mean2) / 39 * 100)

    d = thinkstats2.CohenEffectSize(firsts.prglngth, others.prglngth)
    print('Cohen d', d)


def PrintExtremes(live):
    """Plots the histogram of pregnancy lengths and prints the extremes.

    live: DataFrame of live births
    """
    hist = thinkstats2.Hist(live.prglngth)
    thinkplot.Hist(hist, label='live births')

    thinkplot.Save(root='first_nsfg_hist_live', 
                   title='Histogram',
                   xlabel='weeks',
                   ylabel='frequency')

    print('Shortest lengths:')
    for weeks, freq in hist.Smallest(10):
        print(weeks, freq)

    print('Longest lengths:')
    for weeks, freq in hist.Largest(10):
        print(weeks, freq)
    

def MakeHists(live):
    """Plot Hists for live births

    live: DataFrame
    others: DataFrame
    """
    hist = thinkstats2.Hist(live.birthwgt_lb, label='birthwgt_lb')
    thinkplot.Hist(hist)
    thinkplot.Save(root='first_wgt_lb_hist', 
                   xlabel='pounds',
                   ylabel='frequency',
                   axis=[-1, 14, 0, 3200])

    hist = thinkstats2.Hist(live.birthwgt_oz, label='birthwgt_oz')
    thinkplot.Hist(hist)
    thinkplot.Save(root='first_wgt_oz_hist', 
                   xlabel='ounces',
                   ylabel='frequency',
                   axis=[-1, 16, 0, 1200])

    hist = thinkstats2.Hist(np.floor(live.agepreg), label='agepreg')
    thinkplot.Hist(hist)
    thinkplot.Save(root='first_agepreg_hist', 
                   xlabel='years',
                   ylabel='frequency')

    hist = thinkstats2.Hist(live.prglngth, label='prglngth')
    thinkplot.Hist(hist)
    thinkplot.Save(root='first_prglngth_hist', 
                   xlabel='weeks',
                   ylabel='frequency',
                   axis=[-1, 53, 0, 5000])


def MakeComparison(firsts, others):
    """Plots histograms of pregnancy length for first babies and others.

    firsts: DataFrame
    others: DataFrame
    """
    first_hist = thinkstats2.Hist(firsts.prglngth, label='first')
    other_hist = thinkstats2.Hist(others.prglngth, label='other')

    width = 0.45
    thinkplot.PrePlot(2)
    thinkplot.Hist(first_hist, align='right', width=width)
    thinkplot.Hist(other_hist, align='left', width=width)

    thinkplot.Save(root='first_nsfg_hist', 
                   title='Histogram',
                   xlabel='weeks',
                   ylabel='frequency',
                   axis=[27, 46, 0, 2700])


def main(script):
    live, firsts, others = MakeFrames()

    MakeHists(live)
    PrintExtremes(live)
    MakeComparison(firsts, others)
    Summarize(live, firsts, others)


if __name__ == '__main__':
    import sys
    main(*sys.argv)


