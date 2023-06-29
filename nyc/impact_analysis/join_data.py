import pandas as pd
# import geopandas as gpd
import pandas as pd
# from shapely.geometry import Point
# from geojson import Point, Feature, FeatureCollection, dump
# import requests
# import json
import math
import numpy
import os
import sys
import datetime
import numpy as np

from helpers import map_priority

FSR_columns = ['OBJECTID','SRCategory', 'SRType', 'SRPriority',
       'SRSource', 'SRStatus', 'SRResolution', 'BoroughCode', 'CommunityBoard',
       'ServiceRequestParentGlobalID', 'GlobalID',
       'InitiatedDate', 'ClosedDate', 'CreatedDate', 'UpdatedDate',
       'Descriptor1', 'ComplaintType', 'CallerZipCode', 'SRCallerType',
       'Latitude', 'Longitude', 'Census Tract', 'NTA']#, 'Zip Codes 2']

FI_columns = ['InspectionType', 'InspectionStatus',
       'InspectionTPCondition', 'InspectionTPStructure', 'TreePointDBH',
       'GlobalID', 'ServiceRequestGlobalID', 'InspectionDate', 'ClosedDate', 'CreatedDate',
       'UpdatedDate', 'ParentInspectionGlobalID', 'ReinspectionDate','Location']

FWO_columns = ['WOType', 'WOStatus', 'WOPriority',
       'InspectionGlobalID', 'GlobalID', 'ClosedDate',
       'CancelDate', 'CancelReason','CreatedDate', 'UpdatedDate', 'WOEntity', 
       'PROJSTARTDATE', 'WOProject', 'WOCategory', 'RecommendedSpecies', 'Location', 'ActualFinishDate']

FRA_columns = ['RADefect', 'RADefectLocation', 'Failure', 'ImpactTarget',
       'Consequence', 'RiskRating', 'InspectionGlobalID', 'GlobalID',
       'CreatedDate', 'FailureImpact', 'WorkOrderGlobalID']

cols_to_keep = ['OBJECTID','srcategory', 'complainttype', 'srtype', 'descriptor1', 'srpriority', 'srstatus', 'srresolution', 
        'initiateddate', 'SRClosedDate', 'SRCreatedDate', 'SRUpdatedDate', 'SRGlobalID',
        'inspectiontype', 'inspectionstatus', 'inspectiontpcondition', 'inspectiontpstructure', 
        'inspectiondate', 'InsCreatedDate', 'InsUpdatedDate',  'InsClosedDate', 'reinspectiondate', 'parentinspectionglobalid', 'InsGlobalID',
        'WOGlobalID','wotype', 'wostatus', 'wocategory', 'woentity', 'woproject', 'wopriority',
        'WOCreatedDate', 'WOUpdatedDate',  'WOClosedDate', 'cancelreason', 'canceldate', 'projstartdate',
        'radefect', 'radefectlocation', 'failure', 'failureimpact', 'impacttarget', 'consequence', 'riskrating', 
        'RAGlobalID', 'RACreatedDate', 'nta', 'Borough', 'communityboard', 'zipcode', 'census_tract', 'latitude_SR', 'longitude_SR'] #, 'geometry_SR']


def load_data(directory = './311data_2022/', FSR_columns = FSR_columns, FI_columns = FI_columns, FWO_columns = FWO_columns, FRA_columns = FRA_columns, FSfilename = 'FS.csv', FIfilename = 'FI.csv', FWOfilename = 'FW.csv', FRAfilename = 'FR.csv'):
    FSR = pd.read_csv(f'{directory}/{FSfilename}', usecols=FSR_columns)
    FI = pd.read_csv(f'{directory}/{FIfilename}', usecols=FI_columns)
    FWO = pd.read_csv(f'{directory}/{FWOfilename}', usecols=FWO_columns)
    FRA = pd.read_csv(f'{directory}/{FRAfilename}', usecols=FRA_columns)
    
    print('original lengths: ', len(FSR), len(FI), len(FWO), len(FRA))
    # FSR Global ID corresponds to FI Service Request Global ID
    print('SR in FI not in FSR: ', len(set(FI.ServiceRequestGlobalID) - set(FSR.GlobalID)))
    # FI GLOBAL ID corresponds to FWO Inspection Global ID
    print('IDs in FW not in FI: ', len(set(FWO.InspectionGlobalID) - set(FI.GlobalID)))    
    
    # FRA IncidentGlobalID corresponds to FI Global ID, roughly...
    print(len(set(FRA.InspectionGlobalID) - set(FI.GlobalID)))
    return FSR, FI, FWO, FRA

def preprocess_data(FSR, FI, FWO, FRA):
    FSR.loc[:, "GLOBAL_ID"] = FSR.apply(lambda x:  x.GlobalID if x.ServiceRequestParentGlobalID != x.ServiceRequestParentGlobalID else x.ServiceRequestParentGlobalID, axis = 1)

    FSR.rename(columns={'BoroughCode': 'Borough', 'GLOBAL_ID': 'SRGlobalID', 'ClosedDate': 'SRClosedDate', 'CreatedDate': 'SRCreatedDate', 'UpdatedDate': 'SRUpdatedDate'}, inplace=True)
    FI.rename(columns={'GlobalID': 'InsGlobalID', 'ClosedDate': 'InsClosedDate', 'CreatedDate': 'InsCreatedDate', 'UpdatedDate': 'InsUpdatedDate', 'ServiceRequestGlobalID': 'SRGlobalID'}, inplace=True)
    FWO.rename(columns={'GlobalID': 'WOGlobalID', 'ClosedDate': 'WOClosedDate', 'CreatedDate': 'WOCreatedDate', 'UpdatedDate': 'WOUpdatedDate', 'InspectionGlobalID': 'InsGlobalID'}, inplace=True)
    FRA.rename(columns={'GlobalID': 'RAGlobalID', 'CreatedDate': 'RACreatedDate', 'InspectionGlobalID': 'InsGlobalID'}, inplace=True)

    FSR['Longitude'] = FSR['Longitude'].astype(float)
    FSR['Latitude'] = FSR['Latitude'].astype(float)

    return FSR, FI, FWO, FRA

def merge_data(FSR, FI, FWO, FRA):

    mergeddf = pd.merge(FSR, FI, on='SRGlobalID', how='left', suffixes=('_SR', '_I'), left_index=False, right_index=False)

    #Inspections with SR
    print('Total SRs', len(FSR))
    print('Length of mergeddf', len(mergeddf))
    print('Inspections with SR', len(mergeddf[mergeddf['InsGlobalID'].notna()]))
    print('shapes:', FSR.shape, FI.shape, mergeddf.shape)

    len(mergeddf[mergeddf['InsGlobalID'].isna()])
    
    mergeddf['InsGlobalID'].fillna('-', inplace=True) #so that NAs don't explode the merge
    mergeddf = pd.merge(mergeddf, FWO, on='InsGlobalID', how='left', suffixes=('', '_WO'))
    
    print('shapes after merging FWO:', FWO.shape, mergeddf.shape) #FWO length > mergeddf length, meaning not all work orders have inspections


    # # Merge Risk Assessments
    # merge SRs + Insps + WOs w/ Risk Assessmts
    mergeddf = pd.merge(mergeddf, FRA, on='InsGlobalID', how='left', suffixes=('', '_RA'))
    print('shapes after merging risk assessements: ', FRA.shape, mergeddf.shape)
    
    return mergeddf

def postprocess(df, cols_to_keep = cols_to_keep):
    # df = df[cols_to_keep]
    
    
    # df['initiateddate'] = pd.to_datetime(df['initiateddate'],errors='coerce')
    # df['initiated_year'] = pd.DatetimeIndex(df['initiateddate']).year
    # df['initiated_month'] = pd.DatetimeIndex(df['initiateddate']).month
  
    datecols = ['SRCreatedDate', 'InspectionDate', 'WOClosedDate', 'SRClosedDate','ActualFinishDate']
    for col in datecols:
        df[col] = pd.to_datetime(df[col],errors='coerce')    
    
    df['inspection_attached'] = np.where(df['InsGlobalID']!= '-', True, False)
    df['wo_attached'] = np.where(df['WOGlobalID'].isna(), False, True)
    df['Risk_coded'] = df['RiskRating'].apply(map_priority)
    
    return df

def add_incident_global_ID(
    df, parentcol="ServiceRequestParentGlobalID"
):
    
    df.loc[:, "IncidentGlobalID"] = df.apply(
        lambda x: x[parentcol] if not type(
            x[parentcol]) == float else x["GlobalID"],
        axis=1,
    )
    return df

def pipeline(directory = './311data_2022/', FSR_columns = FSR_columns, FI_columns = FI_columns, FWO_columns = FWO_columns, FRA_columns = FRA_columns, merge_filename = 'joint_public_data'
             , cols_to_keep = cols_to_keep, FSfilename = 'FS.csv', FIfilename = 'FI.csv', FWOfilename = 'FW.csv', FRAfilename = 'FR.csv'):
    combinedfilename = directory + merge_filename + '.csv'
    if not os.path.exists(combinedfilename):
        FSR, FI, FWO, FRA = load_data(directory, FSR_columns, FI_columns, FWO_columns, FRA_columns, FSfilename, FIfilename, FWOfilename, FRAfilename)
        FSR, FI, FWO, FRA = preprocess_data(FSR, FI, FWO, FRA)
        df = merge_data(FSR, FI, FWO, FRA)
        # compression_opts = dict(method='zip', archive_name=f'{merge_filename}.csv')
        df = add_incident_global_ID(df)
        df.to_csv(combinedfilename, index=False) #, compression=compression_opts 
    
    else:
        df = pd.read_csv(combinedfilename)
        
    df = postprocess(df, cols_to_keep)
    return df


