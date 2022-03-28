"""Functions which traverse the data repositories and give the raw acceleration data.
These are implemented as generators, because keeping all of them in memory simultaneously would be prohibitively expensive.
Each file is processed as it arrives, since the final form takes considerably less memory because
we are downsampling by a significant factor and discarding most infomration from most events.
"""

import re
import glob
import itertools
import pandas as pd
from scipy.signal import resample
from progressbar import progressbar


def write_data(path = '/depot/tdm-musafe/data/other_datasets/'):
    """
    Specify the folder in which to write the datasets.
    """
    stream = itertools.chain(
        progressbar(fallalld(), max_value=2515, prefix='FallAllD'),
        progressbar(sis_fall(), max_value=4505, prefix='SisFall'), 
        progressbar(erciyes(), max_value=3326, prefix='Erciyes'),
    )
    all_data = pd.concat(stream)
    all_data.sort_index(inplace = True)
    incidents = all_data['motion'].groupby('incident_id').first()
    acceleration = all_data['x y z'.split()]
    incidents.to_csv(path + 'incidents.csv')
    acceleration.to_csv(path + 'acceleration.csv')

def sis_fall(incident_threshold = 1.775, time_threshold = 4000):
    """Load the SisFall dataset (available from http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/)
    This dataset is published under a creative commons attribution licence, so it should be fine to use in a commercial
    product provided attribution is given.
    
    The default incident threshold and time threshold are based off of the paper at
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8272179/
    """
    cst = re.compile('.*/(.*)\.txt')
    sample_rate = 200
    
    for pathname in glob.iglob('/depot/tdm-musafe/data/other_datasets/SisFall/SisFall_dataset/*/*.txt'):
        acceleration = pd.read_csv(pathname, usecols=[0,1,2], names='x y z'.split(), dtype=int)
        acceleration *= 2**-8
        
        incident_id = 'SisFall_'+cst.match(pathname).group(1)
        code = incident_id.split('_')[1]
        if code.startswith('F'):
            motion = 'fall'
        elif code == 'D18':
            motion = 'trip'
        else:
            motion = 'other'

        yield process_event(acceleration, incident_id, motion, sample_rate, incident_threshold, time_threshold)

        
def erciyes(incident_threshold = 1.33, time_threshold = 4000):
    """Load the Erciyes dataset (available from https://archive.ics.uci.edu/ml/datasets/Simulated+Falls+and+Daily+Living+Activities+Data+Set)
    This dataset is published under a creative commons attribution licence, so it should be fine to use in a commercial
    product provided attribution is given.
    
    The default incident threshold and time threshold are based off of the paper at
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8272179/
    """
    erciyes_testno_person = re.compile(r"/depot/tdm-musafe/data/other_datasets/ErciyesUni/Tests/(.*)/Testler Export/(.*)/340527.txt")
    sample_rate = 25
    
    for path in glob.iglob("/depot/tdm-musafe/data/other_datasets/ErciyesUni/Tests/*/Testler Export/*/*/340527.txt"):
        df = pd.read_csv(path, sep='\t', skiprows=4, usecols=['Acc_X', 'Acc_Y', 'Acc_Z']) / 9.81
        df.columns = ['x', 'y', 'z']

        testnumber, person = erciyes_testno_person.match(path).groups()
        action = person[:3]

        if action in ['808', '809']:
            motion = 'trip'
        elif action[0] == '9':
            motion = 'fall'
        else:
            motion = 'other'

        incident_id = 'erciyes_'+testnumber + '-' + person.replace('/', '-')
        
        event = process_event(df, incident_id, motion,  sample_rate, incident_threshold, time_threshold)
        if not event.isna().any(axis=None):
            yield event
        

def fallalld(incident_threshold = 2.36, time_threshold = 4000):
    """Load the FallAllD dataset (available from https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living)
    This dataset is published under a creative commons attribution licence, so it should be fine to use in a commercial
    product provided attribution is given.
    
    The default incident threshold and time threshold are based off of the paper at
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8272179/
    """    
    fallalld = pd.read_pickle('data/other_datasets/FallALLD/FallAllD.pkl')
    wrists = fallalld[fallalld['Device'] == 'Wrist'].reset_index(drop = True)
    
    sample_rate = 238
    for i, entry in wrists.iterrows():
        subject_id, device, activity_id, trial_no, acc, *rest = entry

        acc = acc * 0.000244

        incident_id = f'FallAllD_S{subject_id}_A{activity_id}_T{trial_no}'

        if activity_id in [101, 102, 121, 122]:
            motion = 'trip'
        elif 103 <= activity_id <= 110 or 123 <= activity_id <= 126:
            motion = 'slip'
        elif 100 <= activity_id <= 135:
            motion = 'fall'
        else:
            motion = 'other'

        yield process_event(acc, incident_id, motion, sample_rate, incident_threshold, time_threshold)

        
def process_event(acceleration, incident_id, motion, sample_rate, incident_threshold, time_threshold):
    """
    Put the acceleration data into a standard form.
    
    By calling pd.concat on the result, we can make a full table without
    having every incident in memory simultaneously. If there is no trigger with
    a wide enough window around it, we still return the data frames but make them empty.
    
    Parameters
    ----------
    acceleration: DataFrame[['x'],['y'],['z']]
        a frame containing the acceleration in its columns. This may include more than one
        incident; the incidents will be separated automatically.
    incident_id: str or int
        the label to attach to the incident. This should be unique, and give enough information
        to find where the incident came from.
    motion: str
        'slip', 'trip', 'fall', or 'other'
    sample_rate: int
        the rate in Hz of the acquisition. This will be used to resample the data to 25Hz.
    incident_threshold:
        because of differences in mass and geometry of the accelerometers, it may make sense
        to choose a different acceleration threshold depending on which accelerometer was used.
        An incident will be returned only when there is a peak with absolute value greater than
        this threshold.
    time_threshold:
        incidents are only counted if the trigger occurs with this much time before and after, in milliseconds.
    
    Returns
    -------
    DataFrame[['x'],['y'],['z'],['motion']]
        doubly indexed by incident_id, then by milliseconds, with 0 corresponding to the incident
        trigger. A suffix is added to incident_id if there is more than one incident.
    """
    if sample_rate == 25:
        a = acceleration
        a.columns = 'x y z'.split()
        a.reset_index(inplace = True, drop = True)
    else:
        a = pd.DataFrame(resample(acceleration, len(acceleration)*25//sample_rate), columns = 'x y z'.split())
    a.index *= 40
    
    triggers = (a.abs() > incident_threshold).any(axis='columns')
    # triggers can't happen at the start or end of the acceleration
    triggers &= a.index > time_threshold
    triggers &= a.index < a.index.max() - time_threshold
    # after a trigger another can't happen until time_threshold has passed
    for i, time in enumerate(a.index):
        if triggers.iloc[i]:
            triggers.loc[(a.index > time) & (a.index <= time + time_threshold)] = False
    
    trig_times = a.index[triggers]
    if len(trig_times) == 0:
        return pd.DataFrame()
    
    windows = []
    for time in trig_times:
        df = a[(time - time_threshold <= a.index) & (a.index <= time + time_threshold)]
        df.index -= time        
        windows.append(df)
    if len(trig_times) == 1:
        incident_ids = [incident_id]
    else:
        incident_ids = [incident_id + f'_{time}' for time in trig_times]
    out = pd.concat(windows, keys = incident_ids, names = ['incident_id', 'milliseconds'])
    out['motion'] = motion
    return out