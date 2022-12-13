import pandas as pd
import numpy as np
import copy
import os

import prepare_data.raw_data_joining as raw_data_joining
import prepare_data.agg_df_creating as cr

from settings import *

"""
To generate the processed dataframe from scratch, do the following:
1. Create a /data/ folder in the root directory
2. Download the following zip file and extract it into /data/: https://1drv.ms/u/s!AihXaAuJppkgoOVLDXazMN1JfBDhjQ?e=TR9aiK (Updated 12/7/2021)
    - This has both public data downloaded from NYC Open Data and private data sent to us by the parks department.
    - It also has the processed data, as of 12/7/2021.
3. Run the pipeline function below

Note: raw forestry data is from: https://data.cityofnewyork.us/Environment/Forestry-Service-Requests/mu46-p9is/data
"""

def pipeline():
    rawdf = raw_data_joining.from_original_data_pipeline()
    return rawdf


def pipeline_with_aggdf(settings_name=settings_default_name, rawdf=None, settings=None, aggdf_folder=aggdf_folder_default, remove_low_categories=True):
    if '{}.csv'.format(settings_name) in os.listdir(aggdf_folder):
        return pd.read_csv(os.path.join(aggdf_folder, '{}.csv'.format(settings_name)))

    # either we're using a dataframe that is already saved, or passing in a settings dictionary to create a new aggdf
    if settings is None and settings_name is not None:
        settings = settings_list.get(settings_name, None)

    assert settings is not None

    if rawdf is None:
        rawdf = pipeline()

    df = cr.prepare_processedrawdf(rawdf)

    if settings.get('incident_limit', None) is not None:
        fewincidents = np.random.choice(
            list(df.IncidentGlobalID.unique()), settings['incident_limit'], replace=False)
        df = df[df.IncidentGlobalID.isin(fewincidents)]
    else:
        df = copy.deepcopy(df)

    aggdf = cr.create_aggdf(df, settings, already_preprocessed=True,
                            remove_low_categories=remove_low_categories)
    return aggdf
