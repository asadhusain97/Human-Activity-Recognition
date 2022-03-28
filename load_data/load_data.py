"""
Load the Makusafe data into a table.

Alden Bradford, January 27 2022
"""


import numpy as np
import pandas as pd
import pickle
from .transformations import batch

dtype = {
    "incident_id": int,
    "sample_number": int,
    "hash_id": "category",
    "motion": "category",
    "miliseconds": int,
    "seconds": int,
    "x": float,
    "y": float,
    "z": float,
}

dates = ["occurrence_ts", "confirmation_ts"]

default_filename = (
    "/depot/tdm-musafe/data/human-activity-recognition/raw-data/har_raw.gz"
)
testing_filename = (
    "/depot/tdm-musafe/data/human-activity-recognition/raw-data/har-test-data-raw.gz"
)

cache_file = "/depot/tdm-musafe/data/cache.pickle"


def load_data(filename=default_filename, *, drop_batches=True, drop_early=False, redo_cache=False, no_cache=False, include_features=False, test_data=False):
    """Load the makusafe data and make them available as DataFrames.

    In order to keep the data in third normal form (without redundant columns),
    the data is split into two frames. By loading the data from scratch whenever
    it is needed, we ensure that the data cannot be corrupted by other processes;
    however, within a process, it is better to keep the frame in memory
    (it should be stored as a variable)

    Parameters
    ----------
    filename: str, optional
        The name of the file from which to load the data.
        
    drop_batches: bool, optional
        if True, give a reduced data set with only those points whose classification
        was not part of a batch.
        
    drop_early: bool, optional
        if True, give a reduced data set with only those points which occurred after
        November 30, 2020. These are missing their time information.
        
    redo_cache: bool, optional
        By default the data is loaded from a cache. If redo_cache is set, then we will write
        to the cache instead.
        
    no_cache: bool, optional
        don't interact with the cache at all.
        
    include_features: bool, optional
        Include the features from the cache, with index aligned to the data chosen.
        
    test_data: bool, optional
        use the test data instead of the training data.

    Returns
    -------
    incidents: DataFrame
    acceleration: DataFrame

    Examples
    --------
    >>> incidents, acceleration = load_data()
    >>> list(incidents)
    ['hash_id', 'motion', 'occurrence_ts', 'confirmation_ts']
    >>> list(acceleration)
    ['x', 'y', 'z']
    >>> list(acceleration.index.names)
    ['incident_id', 'milliseconds']
    """
    if test_data:
        filename = testing_filename
    if filename != default_filename:
        no_cache = True
    if redo_cache or no_cache:
        raw_data = pd.read_csv(filename, dtype=dtype, parse_dates=dates)
        raw_data['milliseconds'] -= 7520
        incidents = (
            raw_data[
                ["hash_id", "motion", "incident_id", "occurrence_ts", "confirmation_ts"]
            ]
            .groupby("incident_id")
            .first()
        )
        acceleration = raw_data[["incident_id", "milliseconds", "x", "y", "z"]].set_index(
            ["incident_id", "milliseconds"]
        )
        earlymask = incidents['occurrence_ts'] > pd.Timestamp('November 30 2020', tz=0)
        batchmask = batch(incidents) == -1
        
        # these dates are unreliable, and should not be used.
        incidents.loc[~earlymask, ['occurrence_ts', 'confirmation_ts']] = np.nan
        
        if not no_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump((incidents, acceleration, earlymask, batchmask), f)
    else:
        with open(cache_file, 'rb') as f:
            incidents, acceleration, earlymask, batchmask = pickle.load(f)
        
    
    # get a true array to use as a mask
    mask = pd.Series(np.ones(len(incidents), dtype=bool), index=incidents.index)
    if drop_early:
        mask &= earlymask
    if drop_batches:
        mask &= batchmask
        
    # we may have lost all the representatives of some categories, so we had better
    # remove those categories
    incidents = incidents[mask]
    for col in 'hash_id', "motion":
        incidents[col] = incidents[col].cat.remove_unused_categories()
    acceleration = acceleration.loc[mask[mask].index]
    if include_features:
        if no_cache:
            # prevents a circular import, and this is not a typical use case
            from .features import make_features
            features = make_features(use_data = acceleration)
        else:
            features = pd.read_csv('/depot/tdm-musafe/data/features.csv', index_col = 'incident_id')[mask]
        return incidents, acceleration, features
    else:
        return incidents, acceleration