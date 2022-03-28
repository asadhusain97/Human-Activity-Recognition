"""
Compute transformations of the data which are not suitable
for use as features, but may be useful to other features.
These are motivated by the lack of gyroscope data; since we
do not have a consistent rotational frame, small changes in
acceleration components are unreliable. The magnitude of the
acceleration is reliable, and the average behavior over long
times is reliable. Hence, we look at only a filtered version
of the direction vector while using an unfiltered acceleration
norm.

Alden Bradford, January 27 2022
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cluster import DBSCAN


def direction(acceleration, frequency_cutoff=0.2, filter_order=8):
    """
    First filter out sharp changes in direction, then normalize to put all directions on the unit sphere.
    Parameters for the filter are available as parameters.

    Parameters
    ----------
    acceleration: DataFrame
        the components of acceleration.

    frequency_cutoff: float, optional
        the frequency (in Hertz) at which we filter. Motions faster than this will not be captured.

    filter_order: int
        the order of the Butterworth filter applied to the signal. A higher order will
        have a bigger effect, but also introduce more artifacts such as ringing.

    Returns
    -------
    direction: DataFrame
        an estimate of the direction of greatest acceleration at each time.
        This is normalized, so x^2+y^2+z^2=1.
    """
    t = len(acceleration.groupby("milliseconds"))
    direction = pd.DataFrame(
        signal.sosfiltfilt(
            sos=signal.butter(filter_order, frequency_cutoff, fs=25, output="sos"),
            x=acceleration.to_numpy().reshape([-1, t, 3]),
            axis=1,
        ).reshape([-1, 3]),
        index=acceleration.index,
        columns=acceleration.columns,
    )
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction


def magnitude(acceleration):
    """
    Find the magnitude of the acceleration.

    Parameters
    ----------
    acceleration: DataFrame
        the components of acceleration.

    Returns
    -------
    Series
        the magnitude of the acceleration.

    Example
    -------
    >>> import pandas as pd
    >>> a = pd.DataFrame([[3, 4, 0],[1, 2, 2]])
    >>> magnitude(a)
    0    5.0
    1    3.0
    Name: acceleration magnitude, dtype: float64
    """
    return pd.Series(
        np.linalg.norm(acceleration, axis=1),
        index=acceleration.index,
        name="acceleration magnitude",
    )


def _batch_employer(incidents, max_gap, min_size, age_threshold):
    """
    This should only be called by batch, which is responsible for ensuring incidents
    from separate employers are not treated as part of the same batch.
    """
    times = incidents['confirmation_ts']
    t = ((times-times.min())/pd.Timedelta(max_gap)).to_numpy().reshape([-1,1])
    # the DBSCAN algorithm does everything we want except for requiring at least one old observation.
    # See its documentation for details.
    labels = DBSCAN(eps=1, min_samples=min_size).fit_predict(t)
    groups = pd.Series(labels, index = times.index, name='batch')
    # remove those batches which do not have an old representative.
    for label in range(labels.max(), -1, -1):
        i = incidents[groups == label]
        age = (i['confirmation_ts']-i['occurrence_ts']).max()
        if age < pd.Timedelta(age_threshold):
            groups[groups == label] = -1
            # shift the labels above this one to fill the gap
            groups[groups > label] -= 1
    return groups


def batch(incidents, max_gap = '5m', min_size = 2, age_threshold = '24h'):
    """
    We can see from the confirmation versus occurrence times that some employers are
    doing their labeling in batches. We should be able to reconstruct this behavior from
    the data.
    
    For these purposes, a batch is a subset of incidents satisfying the following axioms:
    - batches can only contain points from one employer.
    - a batch cannot be a subset of a batch; that is, batches are as large as possible with respect to
      the other axioms.
    - the time between consecutive classifications for a batch must be no more than 5 minutes
    - at least one incident in a batch must be at least 24 hours old at time of classification
    - there must be at least two points in a batch.
    
    This function identifies all batches and assigns each batch an arbitrary number, starting with zero.
    If a point does not belong to a batch, it is given the batch number -1.
    Batch numbers are repeated between employers, so several employers will have batch number 1, for example.
    
    For an explanation of why we would want to treat batched data differently, see:
    /depot/tdm-musafe/etc/batch_demo.ipynb
    """
    return (incidents
        .groupby('hash_id')
        .apply(lambda df: _batch_employer(df, max_gap=max_gap, min_size=min_size, age_threshold=age_threshold))
        .droplevel(0)
        .reindex_like(incidents)
        .rename('batch')
    )

if __name__ == "__main__":
    import doctest

    doctest.testmod()
