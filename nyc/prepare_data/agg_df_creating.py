import pandas as pd
import numpy as np
import prepare_data.agg_df_helpers as dph
import prepare_data.repeat_callers_helpers as adrch


def create_aggregate_df(df, settings):
    """
    Create a dataframe with one row per incident id
    """
    print('Unique incidents in raw data: ', df.IncidentGlobalID.nunique())

    # get death time for incident
    first_report_and_inspection_times = df.groupby("IncidentGlobalID"
                                                   ).agg(**{
                                                         'first_report_datetime': pd.NamedAgg('SRCreatedDate', "min"),
                                                         'first_inspection_datetime': pd.NamedAgg('InspectionDate', "min"),
                                                         'workorder_datetime': pd.NamedAgg('ActualFinishDate', "min"),

                                                         }).reset_index()
    first_report_and_inspection_times.loc[:, 'days_after_first_report'] = first_report_and_inspection_times.first_report_datetime + \
        pd.Timedelta(settings["max_duration"], unit='D')
    first_report_and_inspection_times.loc[:, 'death_time'] = first_report_and_inspection_times[[
        'days_after_first_report', 'first_inspection_datetime', 'workorder_datetime']].min(axis=1)

    first_report_and_inspection_times.loc[:, 'Duration'] = (
        first_report_and_inspection_times.death_time - first_report_and_inspection_times.first_report_datetime).dt.total_seconds() / (60 * 60 * 24)  # Duration in days

    # join death time to df, so that can remove those reports after a death-time
    df = pd.merge(df, first_report_and_inspection_times[[
                  'IncidentGlobalID', 'death_time']], on="IncidentGlobalID")

    print('Number Unique reports before removing reports after incident death: ', df.shape[0])
    df = df[((df.death_time - df.SRCreatedDate).dt.total_seconds()) > 0]
    print('Number Unique reports after removing reports after incident death: ', df.shape[0])

    # this "overlap set" is something used for repeat caller calculation; it is the list of hashes to ignore when checking that a caller hash is duplicated, because these hashes represent missing data. Need to calculate here because the duplicates function below operations on df groups.
    overlapset = adrch.get_overlap_set(df)
    duplicates_per_incident_id = df.groupby('IncidentGlobalID').apply(
        adrch.find_unique_number_duplicates, repeat_caller_conservativeness=settings["repeat_caller_conservativeness"], overlapsetlocal=overlapset).reset_index()
    duplicates_per_incident_id = duplicates_per_incident_id.rename(
        columns={0: 'NumberDuplicates'})

    aggdict = {"SRID": pd.NamedAgg("SRID", "first"),
               "NumberReports": pd.NamedAgg("SRCreatedDate", "count"),
               'Borough': pd.NamedAgg('SRBorough', 'first'),
               'CommunityBoard': pd.NamedAgg('SRCommunityBoard', 'first'),
               'Category': pd.NamedAgg('SRCategory', 'first'),
               'type': pd.NamedAgg('SRtype', 'first'),
               'INSPStructure': pd.NamedAgg('INSPStructure', 'first'),
               'INSPCondtion': pd.NamedAgg('INSPCondtion', 'first'),
               'INSP_RiskAssessment': pd.NamedAgg('INSP_Max_RiskAssment', np.nanmean),
               'WOCategory': pd.NamedAgg('WOCategory', 'first'),
               'WOType': pd.NamedAgg('WOType', 'first'),
               'WORating': pd.NamedAgg('WORating', np.nanmean),
               'WOPriorityCategory': pd.NamedAgg('WOPriorityCategory', 'first'),
               'TPDBH': pd.NamedAgg('TPDBH', np.nanmean),
               'TPStructure': pd.NamedAgg('TPStructure', 'first'),
               'TPCondition': pd.NamedAgg('TPCondition', 'first'),
               'TPSpecies': pd.NamedAgg('TPSpecies', 'first'),
               'SRPriority': pd.NamedAgg('SRPriority', 'first'),
               'TaxClass': pd.NamedAgg('TaxClass', 'first'),
               'ComplaintType': pd.NamedAgg('ComplaintType', 'first'),
               'SRCallerType': pd.NamedAgg('SRCallerType', 'first'),
               'SRCreatedDate': pd.NamedAgg('SRCreatedDate', 'min'),
               'SRClosedDate': pd.NamedAgg('SRClosedDate', 'max'),
               }

    keep_inspection_workorder_datetimes = settings.get('keep_inspection_workorder_datetimes', False)
    if keep_inspection_workorder_datetimes:
        aggdict['first_inspection_datetime'] = pd.NamedAgg('InspectionDate', 'min')
        aggdict['first_workorder_datetime'] = pd.NamedAgg('ActualFinishDate', 'min')

    aggdf = df.groupby("IncidentGlobalID").agg(**aggdict).reset_index()

    aggdf = pd.merge(aggdf, first_report_and_inspection_times[[
        'IncidentGlobalID', 'death_time', 'Duration']], on="IncidentGlobalID")

    aggdf = aggdf.merge(duplicates_per_incident_id, on="IncidentGlobalID")

    print('Unique incidents at end in aggdf: ',
          aggdf.IncidentGlobalID.nunique())

    return aggdf

def merge_with_censustract_labels(df):
    df.loc[:, "SRID"] = df.loc[:, "SRID"].astype(int)
    census_tract_SRID = pd.read_csv("data/SR_to_censustract.csv")
    census_tract_SRID = census_tract_SRID[["SRID", "census_tract_12"]]
    df = pd.merge(df, census_tract_SRID, on="SRID", how="left")
    df = df.rename(columns={"census_tract_12": "census_tract"})
    df.loc[:, "census_tract"] = (
        df.loc[:, "census_tract"].astype(str).str[0:12]
    )
    return df


census_tract_info = None


def load_census_tract_info(tract_file="data/census_organized_all.csv"):
    """
    Loads the census tract data into a pandas dataframe.
    """
    global census_tract_info
    if census_tract_info is None:
        census_tract_info = pd.read_csv(tract_file)
        census_tract_info = census_tract_info.rename(
            columns={"FIPS_BG": "census_tract"}
        )
        census_tract_info.loc[:, "census_tract"] = (
            census_tract_info.loc[:, "census_tract"].astype(str).str[0:12]
        )
    return census_tract_info

def join_census_info_to_df(df):
    census_tract_info = load_census_tract_info()
    # print(census_tract_info.head())
    # print(census_tract_info.info())

    df.loc[:, "census_tract"] = df["census_tract"].astype(str)
    df = df.merge(census_tract_info, on="census_tract", how="left")
    return df


def finalize_aggdf(aggdf, remove_low_categories=False):
    print('finalizing aggdf, unique incidents starting: ',
          aggdf.IncidentGlobalID.nunique())
    
    if 'avg_household_income' in aggdf.columns:
        aggdf.loc[:, 'loghouseholdincome'] = aggdf.eval(
            'log(avg_household_income)')
    else:
        aggdf.loc[:, 'loghouseholdincome'] = aggdf.eval(
                'log(median_household_income)')

    aggdf.loc[:, 'logdensity'] = aggdf.eval('log(density + 1)')
    aggdf.loc[:, 'logTPDBH'] = aggdf.eval('log(TPDBH+1)')
    if remove_low_categories:
        aggdf = aggdf[~aggdf.Category.isin(
            ['Rescue/Preservation', 'Pest/Disease', 'Remove Stump', 'Planting Space', 'Other'])]
        print('unique incidents after removing low categories: ',
              aggdf.IncidentGlobalID.nunique())

    aggdf = aggdf.query('TPDBH<100')
    print('unique incidents after removing big trees: ',
          aggdf.IncidentGlobalID.nunique())
    aggdf = aggdf.query('Duration>.1')
    print('unique incidents after removing short duration: ',
          aggdf.IncidentGlobalID.nunique())
    if 'logdensity' in aggdf.columns:
        aggdf = aggdf.query('logdensity>=0')
        print('unique incidents after removing ngative log density: ',
          aggdf.IncidentGlobalID.nunique())

    if 'WOPriorityCategory' in aggdf.columns:
        aggdf.loc[:, 'WOPriorityCategory'] = aggdf.loc[:,
                                                 'WOPriorityCategory'].replace({'A': 'A_B', 'B': 'A_B'})
    if 'INSPCondtion' in aggdf.columns:
        aggdf.loc[:, 'INSPCondtion'] = aggdf.loc[:, 'INSPCondtion'].replace(
            {'Excellent': 'Excellent_Good', 'Good': 'Excellent_Good', 'Critical': 'Critical_Poor', 'Poor': 'Critical_Poor'})
        aggdf = aggdf.query('INSPCondtion !="Unknown"')
        
    return aggdf


def prepare_processedrawdf(rawdf):
    rawdf = dph.filter_inspected(rawdf)
    rawdf = dph.convert_SR_cols_to_datetime(rawdf)
    return rawdf


def create_aggdf(rawdf, settings, already_preprocessed=False, remove_low_categories=False):
    """
    Preprocessing pipeline for the estimation data.
    """

    if not already_preprocessed:
        rawdf = prepare_processedrawdf(rawdf)

    aggdf = create_aggregate_df(rawdf, settings)

    aggdf = merge_with_censustract_labels(aggdf)
    aggdf = join_census_info_to_df(aggdf)

    aggdf = finalize_aggdf(aggdf, remove_low_categories=remove_low_categories)

    return aggdf