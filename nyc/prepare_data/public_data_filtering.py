from venv import create
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from cmath import nan

import random
import gc

from os.path import exists

default_settings = {'max_duration': 100, 'sample_size': -1}
thirty_settings = {'max_duration': 30, 'sample_size': -1}
two_hundred_settings = {'max_duration': 200, 'sample_size': -1}


def create_aggregated_df_from_public(settings = default_settings, FSR_file = 'data/FSR_221022.csv', FI_file = 'data/FI_221022.csv',
                         FWO_file = 'data/FWO_221022.csv',
                         FRA_file = 'data/FRA_221024.csv', shape_file = 'visualize/viz_data/nycb2020.shp'):
    """
    create the aggregated dataframe from the following settings:
       max_duration: used to truncate the observation interval
       sample_size: number of unique **incidents** to sample from the raw data; if -1, keep all data
    """
    aggdf_file = 'data_clean/aggdf_max{}_samp{}.csv'.format(settings['max_duration'], settings['sample_size']) 
    if exists(aggdf_file):
        print('{} already exists'.format(aggdf_file))
        aggdf = pd.read_csv(aggdf_file, dtype = {'census_tract': str})
    else:
        print('{} does not exist, reading raw data'.format(aggdf_file))
        FSR = pd.read_csv(FSR_file)
        FI = pd.read_csv(FI_file)
        FWO = pd.read_csv(FWO_file)
        FRA = pd.read_csv(FRA_file)
        shape = gpd.read_file(shape_file)

        shape.GEOID = shape.GEOID.astype(str).str[0:12]
        shape = shape.to_crs({'init': 'epsg:4326'}) ## convert to lat/long

        # assign a global ID for service requests
        FSR.loc[:, "IncidentGlobalID"] = FSR.apply(lambda x:  x.GlobalID if x.ServiceRequestParentGlobalID!= x.ServiceRequestParentGlobalID else x.ServiceRequestParentGlobalID, axis = 1)

        print("# incidents in FSR, before filtering", FSR.IncidentGlobalID.nunique())
        
        # sample the data by IncidentGlobalID
        if settings['sample_size'] > 0:
            id_list = FSR.IncidentGlobalID.unique()
            sample_id = random.sample(list(id_list), settings['sample_size'])
            FSR = FSR.query('IncidentGlobalID in @sample_id')
            print("Number of reports after sampling", FSR.shape[0])
        
        # get the census tract (block here) for each SR
        geometry = [Point(xy) for xy in zip(FSR.Longitude, FSR.Latitude)]
        crs = {'init' :'epsg:2263'} # this is for originally lat/long 
        gdf = gpd.GeoDataFrame(FSR, crs=crs, geometry=geometry)
        merged_file = gpd.sjoin(gdf, shape, how='left', op='within')
        FSR = pd.DataFrame(merged_file)


        # merge the inspection, work order, and service request data
        FSR.rename(columns={'OBJECTID': 'SRID', 'GlobalID': 'SRGlobalID', 'ClosedDate': 'SRClosedDate', 'CreatedDate': 'SRCreatedDate', 'UpdatedDate': 'SRUpdatedDate', 'CommunityBoard': 'SRCommunityBoard'}, inplace=True)
        FI.rename(columns={'GlobalID': 'InsGlobalID', 'ClosedDate': 'InsClosedDate', 'CreatedDate': 'InsCreatedDate', 'UpdatedDate': 'InsUpdatedDate', 'ServiceRequestGlobalID': 'SRGlobalID'}, inplace=True)
        FWO.rename(columns={'GlobalID': 'WOGlobalID', 'ClosedDate': 'WOClosedDate', 'CreatedDate': 'WOCreatedDate', 'UpdatedDate': 'WOUpdatedDate', 'InspectionGlobalID': 'InsGlobalID'}, inplace=True)
        FRA.rename(columns={'GlobalID': 'RAGlobalID', 'ClosedDate': 'RAClosedDate', 'CreatedDate': 'RACreatedDate', 'InspectionGlobalID': 'InsGlobalID'}, inplace=True)

        firstmerge = pd.merge(FSR, FI, on='SRGlobalID', how='left')
        print('SR-I merge length', firstmerge.shape[0])
        firstmerge['InsGlobalID'].fillna('0', inplace=True)
        # firstmerge.loc[:, 'Inspected'] = 1-firstmerge['InsGlobalID'].isna()
        # inc_inspected = firstmerge.groupby('IncidentGlobalID').agg({'Inspected': 'sum'}).reset_index()
        # inc_inspected = inc_inspected.query('Inspected > 0')
        # print('# incidents with inspection', inc_inspected.shape[0])
        # firstmerge = firstmerge.query('IncidentGlobalID in @inc_inspected.IncidentGlobalID')


        # here we need to filter out the uninspected incidents
        inspected_incidents = firstmerge.query('InsGlobalID != "0"').IncidentGlobalID.unique()
        uninspected = firstmerge.query('InsGlobalID == "0"')
        uninspected.to_csv('data_clean/uninspectedSR.csv', index = False)
        print('# inspected incidents', len(inspected_incidents))
        firstmerge = firstmerge.query('IncidentGlobalID in @inspected_incidents')
        print('# reports belonging to inspected incidents', firstmerge.shape[0])

        secondmerge = pd.merge(firstmerge, FWO, on='InsGlobalID', how='left')

        print("done merging SR-I-WO")

        del FSR, FI, FWO, firstmerge, gdf, merged_file, shape, geometry, crs
        gc.collect()

        secondmerge = pd.merge(secondmerge, FRA, on='InsGlobalID', how='left')

        print("done merging SR-I-WO-RA")

        del FRA
        gc.collect()

        # secondmerge.to_csv('data_clean/secondmerge_{}_{}.csv'.format(settings['max_duration'], settings['sample_size']), index = False)
        
        # then actually do the cleaning
        aggdf = secondmerge.copy()
        del secondmerge
        gc.collect()
        # remove one anomalous data point
        internal_data_sources = [
        'AMPS', 'DPR', 'DOT', 'Park Inspection Program', 'FDNY'
        ] # reports by non-public sources are also in the data; remove them.
    
        aggdf = aggdf[~aggdf.SRSource.isin(internal_data_sources)]
        
        print("# reports, before cleaning: SR-I-WO-RA merge length", aggdf.shape[0])
        aggdf.loc[:, "SRCreatedDate"] = pd.to_datetime(aggdf.SRCreatedDate, errors = 'coerce')
        aggdf.loc[:, "InspectionDate"] = pd.to_datetime(aggdf.InspectionDate, errors = 'coerce')
        aggdf.loc[:, 'ActualFinishDate'] = pd.to_datetime(aggdf.ActualFinishDate, errors = 'coerce')
        aggdf.loc[:, 'SRClosedDate'] = pd.to_datetime(aggdf.SRClosedDate, errors = 'coerce')

        # filter out the handful of requests at the very beginning and very end
        first_day = pd.to_datetime('2015-03-01 00:00:00')
        last_day = pd.to_datetime('2022-08-31 23:59:59')
        aggdf = aggdf.query('SRCreatedDate >= @first_day')
        aggdf = aggdf.query('SRCreatedDate <= @last_day')

        # dfreports = aggdf.copy()

        print('first report time ', aggdf.SRCreatedDate.min())
        print('last report time ', aggdf.SRCreatedDate.max())

        first_report_and_last_modified_times = aggdf.groupby('IncidentGlobalID').agg(**{
        "first_report_datetime": pd.NamedAgg('SRCreatedDate', 'min'),
        # "last_modified_datetime": pd.NamedAgg('LAST_MODIFIED_DATE', 'max'),
        "first_inspection_datetime": pd.NamedAgg('InspectionDate', 'min'),
        "actual_finish_date": pd.NamedAgg('ActualFinishDate', 'min')
            }).reset_index()
        

        first_report_and_last_modified_times.loc[:,"days_after_first_report"] = \
            first_report_and_last_modified_times.first_report_datetime + pd.Timedelta(days = settings['max_duration'])
        
        first_report_and_last_modified_times.loc[:, 'last_day_in_dataset'] = pd.to_datetime("9/2/2022 00:00")
        
        # calculate death time; potentially something wrong here, since we are filtering out a lot of requests
        first_report_and_last_modified_times.loc[:,"death_time"] = \
            first_report_and_last_modified_times[['first_inspection_datetime', 'actual_finish_date', 'days_after_first_report', 'last_day_in_dataset']].min(axis = 1)


        first_report_and_last_modified_times.loc[:, 'Duration'] = \
            (first_report_and_last_modified_times.death_time - first_report_and_last_modified_times.first_report_datetime).dt.total_seconds() / (60 * 60 *24)
        
        aggdf = \
            pd.merge(aggdf, first_report_and_last_modified_times[['IncidentGlobalID','death_time']], on = 'IncidentGlobalID')

        aggdf = aggdf.query("death_time > SRCreatedDate")

        twothousands = pd.to_datetime('2000-01-01 00:00:00')
        aggdf = aggdf.query("death_time > @twothousands")

        print('first death time ', aggdf.death_time.min())
        print('last death time ', aggdf.death_time.max())
        

        print("# reports, after filtering out created after death", aggdf.shape[0])

        aggdict = {"SRID": pd.NamedAgg("SRID", "first"),
               "NumberReports": pd.NamedAgg("SRCreatedDate", "nunique"), # here use nunique to bypass the problem of one request triggers several inspections
               'Borough': pd.NamedAgg('BoroughCode', 'first'),
               'CommunityBoard': pd.NamedAgg('CommunityBoard', 'first'),
               'Category': pd.NamedAgg('SRCategory', 'first'),
               'type': pd.NamedAgg('SRType', 'first'),
               'INSPStructure': pd.NamedAgg('InspectionTPStructure', 'first'),
               'INSPCondtion': pd.NamedAgg('InspectionTPCondition', 'first'),
               'INSP_RiskAssessment': pd.NamedAgg('RiskRating', np.nanmean),
               'WOCategory': pd.NamedAgg('WOCategory', 'first'),
               'WOType': pd.NamedAgg('WOType', 'first'),
            #    'WORating': pd.NamedAgg('WORating', np.nanmean),
               'WOPriorityCategory': pd.NamedAgg('WOPriority', 'first'),
               'TPDBH': pd.NamedAgg('TreePointDBH', np.nanmean),
            #    'TPStructure': pd.NamedAgg('TPStructure', 'first'),
            #    'TPCondition': pd.NamedAgg('TPCondition', 'first'),
            #    'TPSpecies': pd.NamedAgg('TPSpecies', 'first'),
               'SRPriority': pd.NamedAgg('SRPriority', 'first'),
               'TaxClass': pd.NamedAgg('TaxClass', 'first'),
               'ComplaintType': pd.NamedAgg('ComplaintType', 'first'),
               'SRCallerType': pd.NamedAgg('SRCallerType', 'first'),
               'SRCreatedDate': pd.NamedAgg('SRCreatedDate', 'min'),
               'SRClosedDate': pd.NamedAgg('SRClosedDate', 'max'),
               'census_tract': pd.NamedAgg('GEOID', 'first'),
        }

        aggdf = aggdf.groupby("IncidentGlobalID").agg(**aggdict).reset_index()

        aggdf = \
            pd.merge(aggdf, first_report_and_last_modified_times[['IncidentGlobalID','Duration']], on = 'IncidentGlobalID')

        aggdf.loc[:,'NumberDuplicates'] = aggdf.NumberReports - 1

        # aggdf.loc[:,'SRCreatedDate'] = aggdf.SRCreatedDate.dt.to_period('M')
        # aggdf = aggdf.sort_values(by = 'SRCreatedDate').reset_index()
        # aggdf.SRCreatedDate = aggdf.SRCreatedDate.astype('str')

        print("# incidents, after filtering out created after death", aggdf.shape[0])

        aggdf = aggdf.query('Duration > 0.1')

        print('# incidents, filtering out extremely short duration <=0.1days', aggdf.shape[0])

        aggdf.loc[:,'logTPDBH'] = aggdf.eval('log(TPDBH+1)')
        aggdf = aggdf.query('TPDBH<100')
        print('# incidents, filtering out extremely large DBH', aggdf.shape[0])

        aggdf = aggdf[~aggdf.Category.isin(
            ['Rescue/Preservation', 'Pest/Disease', 'Remove Stump', 'Planting Space', 'Other', 'Claims', 'Plant Tree', 'Remove Debris'])]
        if 'INSPCondtion' in aggdf.columns:
            aggdf.loc[:, 'INSPCondtion'] = aggdf.loc[:, 'INSPCondtion'].replace(
                {'Excellent': 'Excellent_Good', 'Good': 'Excellent_Good', 'Critical': 'Critical_Poor', 'Poor': 'Critical_Poor'})
            aggdf = aggdf.query('INSPCondtion !="Unknown"')
        print('# incidents after removing low categories: ', aggdf.shape[0])

        print('# reports, represented in the final aggdf', aggdf.NumberReports.sum())

        aggdf.to_csv(aggdf_file, index = False)

    return aggdf




def add_census_demographics(aggdf, settings, census_attributes_file = 'data/census_organized_all.csv'):
    demographics = pd.read_csv(census_attributes_file, dtype = {'FIPS_BG': str})
    demographics.rename(columns = {'FIPS_BG': 'census_tract'}, inplace = True)
    demographics.loc[:, 'loghouseholdincome'] = demographics.eval("log(avg_household_income + 1)")
    demographics.loc[:, 'logdensity'] = demographics.eval("log(density + 1)")

    aggdf.census_tract = aggdf.census_tract.astype(str)

    mergeddf = pd.merge(aggdf, demographics, on = 'census_tract')
    mergeddf = mergeddf.query('loghouseholdincome > 0')
    mergeddf = mergeddf.query('logdensity > 0')

    print("min, max log density", mergeddf.logdensity.min(), mergeddf.logdensity.max())
    print("min, max log income", mergeddf.loghouseholdincome.min(), mergeddf.loghouseholdincome.max())

    print("# incidents, after merging with census demographics and dropping negative density and income", mergeddf.shape[0])

    mergeddf.to_csv('data_clean/aggdf_max{}_samp{}_census_demographics.csv'.format(settings['max_duration'], settings['sample_size']), index = False)
    return mergeddf

def public_data_pipeline_with_settings(settings = default_settings):
    """
    create the aggregated dataframe from the following settings:
       department: whether filter by department, if True, filter only CDOT
       completed: whether filter by completed
       max_duration: used to truncate the observation interval
    """
    aggdf = create_aggregated_df_from_public(settings)
    # aggdf_with_census = add_census_tracts(aggdf, settings)
    aggdf_with_demo = add_census_demographics(aggdf, settings)
    return aggdf, aggdf_with_demo



def create_dfreports_public(settings = default_settings, FSR_file = 'data/FSR_221022.csv', FI_file = 'data/FI_221022.csv',
                         FWO_file = 'data/FWO_221022.csv',
                         FRA_file = 'data/FRA_221024.csv', shape_file = 'visualize/viz_data/nycb2020.shp'):
    """
    create the dfreports
    """
    FSR = pd.read_csv(FSR_file)
    FI = pd.read_csv(FI_file)
    FWO = pd.read_csv(FWO_file)
    FRA = pd.read_csv(FRA_file)

    srcategories=[
        "Hazard",
        "Root/Sewer/Sidewalk",
        "Illegal Tree Damage",
        "Remove Tree",
        "Remove Stump",
        "Prune",
        "Rescue/Preservation",
        "Pest/Disease",
        "Planting Space"
    ]



    # assign a global ID for service requests
    FSR.loc[:, "IncidentGlobalID"] = FSR.apply(lambda x:  x.GlobalID if x.ServiceRequestParentGlobalID!= x.ServiceRequestParentGlobalID else x.ServiceRequestParentGlobalID, axis = 1)

    FSR = FSR[FSR.SRCategory.isin(srcategories)]

    print("# incidents in FSR, before filtering", FSR.IncidentGlobalID.nunique())
    


    # merge the inspection, work order, and service request data
    FSR.rename(columns={'OBJECTID': 'SRID', 'GlobalID': 'SRGlobalID', 'ClosedDate': 'SRClosedDate', 'CreatedDate': 'SRCreatedDate', 'UpdatedDate': 'SRUpdatedDate', 'CommunityBoard': 'SRCommunityBoard'}, inplace=True)
    FI.rename(columns={'GlobalID': 'InsGlobalID', 'ClosedDate': 'InsClosedDate', 'CreatedDate': 'InsCreatedDate', 'UpdatedDate': 'InsUpdatedDate', 'ServiceRequestGlobalID': 'SRGlobalID'}, inplace=True)
    FWO.rename(columns={'GlobalID': 'WOGlobalID', 'ClosedDate': 'WOClosedDate', 'CreatedDate': 'WOCreatedDate', 'UpdatedDate': 'WOUpdatedDate', 'InspectionGlobalID': 'InsGlobalID'}, inplace=True)
    FRA.rename(columns={'GlobalID': 'RAGlobalID', 'ClosedDate': 'RAClosedDate', 'CreatedDate': 'RACreatedDate', 'InspectionGlobalID': 'InsGlobalID'}, inplace=True)

    firstmerge = pd.merge(FSR, FI, on='SRGlobalID', how='left')
    print('SR-I merge length', firstmerge.shape[0])
    firstmerge['InsGlobalID'].fillna('0', inplace=True)


    # here we need to filter out the uninspected incidents
    inspected_incidents = firstmerge.query('InsGlobalID != "0"').IncidentGlobalID.unique()

    firstmerge.loc[:, 'Inspected'] = firstmerge.eval('IncidentGlobalID in @inspected_incidents')

    secondmerge = pd.merge(firstmerge, FWO, on='InsGlobalID', how='left')

    secondmerge = secondmerge.drop_duplicates(subset='SRID', keep="first") # here need to do a drop

    print("done merging SR-I-WO")

    del FSR, FI, FWO, firstmerge
    gc.collect()

    secondmerge = pd.merge(secondmerge, FRA, on='InsGlobalID', how='left')

    print("done merging SR-I-WO-RA")

    del FRA
    gc.collect()

    secondmerge.loc[:, "SRCreatedDate"] = pd.to_datetime(secondmerge.SRCreatedDate, errors = 'coerce')
    secondmerge.loc[:, "InspectionDate"] = pd.to_datetime(secondmerge.InspectionDate, errors = 'coerce')
    secondmerge.loc[:, 'ActualFinishDate'] = pd.to_datetime(secondmerge.ActualFinishDate, errors = 'coerce')
    secondmerge.loc[:, 'SRClosedDate'] = pd.to_datetime(secondmerge.SRClosedDate, errors = 'coerce')

    collapsecategory = ['Remove Stump', 'Rescue/Preservation', 'Pest/Disease', 'Planting Space', 'Claims']
    secondmerge.loc[secondmerge.SRCategory.isin(collapsecategory) ,'SRCategory'] = 'Other'

    internal_data_sources = [
        'AMPS', 'DPR', 'DOT', 'Park Inspection Program', 'FDNY'
        ] # reports by non-public sources are also in the data; remove them.
    
    secondmerge = secondmerge[~secondmerge.SRSource.isin(internal_data_sources)]

    return secondmerge