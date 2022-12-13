from venv import create
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import random

from os.path import exists

default_settings = {'department': 'CDOTDWM', 'completed': False, 'max_duration': 100, 'sample_size': -1}
thirty_settings = {'department': 'CDOTDWM', 'completed': False, 'max_duration': 30, 'sample_size': -1}
two_hundred_settings = {'department': 'CDOTDWM', 'completed': False, 'max_duration': 200, 'sample_size': -1}


def create_aggregated_df(settings, raw_data_file = 'data/311_Service_Requests.csv'):
    """
    create the aggregated dataframe from the following settings:
       department: whether filter by department, if True, filter only CDOT
       completed: whether filter by completed
       max_duration: used to truncate the observation interval
       sample_size: number of unique **incidents** to sample from the raw data; if 0, keep all data
    """
    aggdf_file = 'data_clean/aggdf_dep{}_comp{}_max{}_samp{}.csv'.format(settings['department'], settings['completed'], settings['max_duration'], settings['sample_size']) 
    if exists(aggdf_file):
        print('{} already exists'.format(aggdf_file))
        aggdf = pd.read_csv(aggdf_file, dtype = {'census_tract': str})
    else:
        print('{} does not exist, reading raw data'.format(aggdf_file))
        raw_data = pd.read_csv(raw_data_file)


        # filter by department
        if settings['department'] == 'CDOT':
            aggdf = raw_data.loc[raw_data.OWNER_DEPARTMENT == 'CDOT - Department of Transportation']
            aggdf = aggdf.replace({'OWNER_DEPARTMENT': {'CDOT - Department of Transportation': 'CDOT'}})
            print("Number of unique reports owned by CDOT", aggdf.shape[0])
            # filter out PCL3 and VBL which have no duplicates
            aggdf = aggdf.query("SR_SHORT_CODE not in ['PCL3', 'VBL']")
            print("Number of unique reports owned by CDOT, after filtering types with no duplicates", aggdf.shape[0])

        elif settings['department'] == 'CDOTDWM':
            aggdf = raw_data.loc[(raw_data.OWNER_DEPARTMENT == 'CDOT - Department of Transportation')|(raw_data.OWNER_DEPARTMENT == 'DWM - Department of Water Management')]
            aggdf = aggdf.replace({'OWNER_DEPARTMENT': {'CDOT - Department of Transportation': 'CDOT', 'DWM - Department of Water Management': 'DWM'}})
            print("Number of reports owned by CDOT and DWM", aggdf.shape[0])
            aggdf = aggdf.query("SR_SHORT_CODE not in ['PCL3', 'VBL']")
            print("Number of unique reports owned by CDOT and DWM, after filtering types with no duplicates", aggdf.shape[0])
        else:
            aggdf = raw_data
        
        # filter by completed
        if settings['completed'] == True:
            aggdf = aggdf.loc[aggdf.STATUS == 'Completed']
            print("Number of unique reports that are completed", aggdf.shape[0])

        # assign GLOBAL_ID
        aggdf.loc[:,"GLOBAL_ID"] = aggdf.apply(lambda x:  x.SR_NUMBER if x.DUPLICATE == False else x.PARENT_SR_NUMBER, axis = 1)
        
        # sample the data by GLOBAL_ID
        if settings['sample_size'] > 0:
            id_list = aggdf.GLOBAL_ID.unique()
            sample_id = random.sample(list(id_list), settings['sample_size'])
            aggdf = aggdf.query('GLOBAL_ID in @sample_id')
            print("Number of unique reports after sampling", aggdf.shape[0])
        
        # then actually do the cleaning
        aggdf.loc[:, "CREATED_DATE"] = pd.to_datetime(aggdf.CREATED_DATE)
        # aggdf.loc[:, "LAST_MODIFIED_DATE"] = pd.to_datetime(aggdf.LAST_MODIFIED_DATE)
        aggdf.loc[:, "CLOSED_DATE"] = pd.to_datetime(aggdf.CLOSED_DATE)

        aggdf = aggdf.query("CREATED_DATE >= '2019-03-01'")
        # print("Number of unique reports after filtering by date", aggdf.shape[0])

        print('first report time ', aggdf.CREATED_DATE.min())
        print('last report time ', aggdf.CREATED_DATE.max())

        first_report_and_last_modified_times = aggdf.groupby('GLOBAL_ID').agg(**{
        "first_report_datetime": pd.NamedAgg('CREATED_DATE', 'min'),
        # "last_modified_datetime": pd.NamedAgg('LAST_MODIFIED_DATE', 'max'),
        "closed_datetime": pd.NamedAgg('CLOSED_DATE', 'min'),
            }).reset_index()

        first_report_and_last_modified_times.loc[:,"days_after_first_report"] = \
            first_report_and_last_modified_times.first_report_datetime + pd.Timedelta(days = settings['max_duration'])
        
        first_report_and_last_modified_times.loc[:, 'last_day_in_dataset'] = pd.to_datetime("7/4/2022 21:20")
        
        # first_report_and_last_modified_times.loc[:,"death_time"] = \
            # first_report_and_last_modified_times[['closed_datetime', 'last_modified_datetime', 'days_after_first_report']].min(axis = 1)
        first_report_and_last_modified_times.loc[:,"death_time"] = \
            first_report_and_last_modified_times[['closed_datetime', 'last_day_in_dataset', 'days_after_first_report']].min(axis = 1)
        # first_report_and_last_modified_times.loc[:,"death_time"] = \
            # first_report_and_last_modified_times[['closed_datetime', 'days_after_first_report']].min(axis = 1)

        print('first death time ', first_report_and_last_modified_times.death_time.min())
        print('last death time ', first_report_and_last_modified_times.death_time.max())
        

        first_report_and_last_modified_times.loc[:, 'Duration'] = \
            (first_report_and_last_modified_times.death_time - first_report_and_last_modified_times.first_report_datetime).dt.total_seconds() / 60
        
        aggdf = \
            pd.merge(aggdf, first_report_and_last_modified_times[['GLOBAL_ID','death_time']], on = 'GLOBAL_ID')
        aggdf = aggdf.query("death_time > CREATED_DATE")

        print("Number of unique reports, after truncation", aggdf.shape[0])

        aggdict = {
        "NumberReports": pd.NamedAgg('CREATED_DATE', 'count'),
        "CreatedDate": pd.NamedAgg('CREATED_DATE', 'first'),
        "LATITUDE": pd.NamedAgg('LATITUDE', 'first'),
        "LONGITUDE": pd.NamedAgg('LONGITUDE', 'first'),
        "SR_TYPE": pd.NamedAgg('SR_TYPE', 'first'),
        "SR_SHORT_CODE": pd.NamedAgg('SR_SHORT_CODE', 'first'),
        "OWNER_DEPARTMENT": pd.NamedAgg('OWNER_DEPARTMENT', 'first'),
        "STATUS": pd.NamedAgg('STATUS', 'first')
        } 

        aggdf = aggdf.groupby("GLOBAL_ID").agg(**aggdict).reset_index()

        aggdf = \
            pd.merge(aggdf, first_report_and_last_modified_times[['GLOBAL_ID','Duration']], on = 'GLOBAL_ID')

        ## transform to days instead of minutes
        aggdf.Duration = aggdf.Duration/1440

        aggdf.loc[:,'NumberDuplicates'] = aggdf.NumberReports - 1
        aggdf = aggdf.dropna()

        aggdf.loc[:,'CreatedDate'] = aggdf.CreatedDate.dt.to_period('M')
        aggdf = aggdf.sort_values(by = 'CreatedDate').reset_index()
        aggdf.CreatedDate = aggdf.CreatedDate.astype('str')

        print("Number of unique incidents, after truncation", aggdf.shape[0])

        aggdf = aggdf.query('Duration > 0.01')

        print('Number of unique incidents, after filtering out extremely short duration', aggdf.shape[0])
        print('Number of unique reports, represented', aggdf.NumberReports.sum())

        aggdf.to_csv(aggdf_file, index = False)

    return aggdf

def add_census_tracts(aggdf, settings, sr_to_census_file = 'data/SR_to_census_tracts_full.csv'):
    census_tracts = pd.read_csv(sr_to_census_file, dtype={"census_tract": str})
    mergeddf = pd.merge(aggdf, census_tracts[['GLOBAL_ID','census_tract']], on = 'GLOBAL_ID')
    mergeddf = mergeddf.drop(['LATITUDE', 'LONGITUDE'], axis = 1).dropna().reset_index(drop = True)
    print("Number of unique incidents, after merging with census tracts and dropping nans", mergeddf.shape[0])

    # mergeddf.census_tract = mergeddf.census_tract.str[:12]
    ## the retrieved census tract ids dont perfectly match the ones in the census demo data
    mergeddf.loc[:, "census_tract12"] = mergeddf.census_tract.str[:12]

    mergeddf.to_csv('data_clean/aggdf_dep{}_comp{}_max{}_samp{}_census.csv'.format(settings['department'], settings['completed'], settings['max_duration'], settings['sample_size']), index = False)
    return mergeddf

def add_census_demographics(aggdf, settings, census_attributes_file = 'data/census_organized_all.csv'):
    demographics = pd.read_csv(census_attributes_file, dtype = {'FIPS_BG': str})
    demographics.rename(columns = {'FIPS_BG': 'census_tract12'}, inplace = True)
    demographics.loc[:, 'avg_household_income'] = np.log(demographics.avg_household_income)
    demographics.loc[:, 'density'] = np.log(demographics.density)

    mergeddf = pd.merge(aggdf, demographics, on = 'census_tract12')
    print("Number of unique incidents, after merging with census demographics", mergeddf.shape[0])

    mergeddf.to_csv('data_clean/aggdf_dep{}_comp{}_max{}_samp{}_census_demographics.csv'.format(settings['department'], settings['completed'], settings['max_duration'], settings['sample_size']), index = False)
    return mergeddf

def pipeline_with_settings(settings = default_settings):
    """
    create the aggregated dataframe from the following settings:
       department: whether filter by department, if True, filter only CDOT
       completed: whether filter by completed
       max_duration: used to truncate the observation interval
    """
    aggdf = create_aggregated_df(settings)
    aggdf_with_census = add_census_tracts(aggdf, settings)
    aggdf_with_demo = add_census_demographics(aggdf_with_census, settings)
    return aggdf, aggdf_with_census, aggdf_with_demo