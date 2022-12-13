import pandas as pd
import datetime
import os
import numpy as np

import prepare_data.agg_df_helpers as dph


def filter_public_df(
    dfpublic,
    start_date="2017-06-30",
    end_date="2020-07-01",
    srcategories=[
        "Hazard",
        "Root/Sewer/Sidewalk",
        "Illegal Tree Damage",
        "Remove Tree",
        "Remove Stump",
        "Prune",
        "Rescue/Preservation",
        "Pest/Disease",
        "Planting Space",
    ],
):
    """
    Join public 311 and private parks department datasets.
    """
    print("Filtering public data...")
    dfpublic = dfpublic.rename(columns={"OBJECTID": "SRID"})
    
    # Filter to only the service requests that the parks department sent; Not tree planting
    dfpublic = dfpublic[dfpublic.SRCategory.isin(srcategories)]
    collapsecategory = ['Remove Stump', 'Rescue/Preservation', 'Pest/Disease', 'Planting Space']
    dfpublic.loc[dfpublic.SRCategory.isin(collapsecategory) ,'SRCategory'] = 'Other'
    # print(dfpublic.SRCategory.value_counts())
    
    dfpublic.loc[:, "CreatedDate"] = pd.to_datetime(dfpublic["CreatedDate"])

    lower = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    higher = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    dfpublic_filtered = dfpublic.query(
        "CreatedDate >= @lower and CreatedDate < @higher"
    )

    return dfpublic_filtered


def join_public_private(dfpublic_filtered, dfparks):
    print("Joining public and private datasets...")
    srids_parks = set(dfparks.SRID)
    srids_public_filtered = set(dfpublic_filtered.SRID)
    print(
        "SRIDs that are not in each others:",
        len(srids_parks - srids_public_filtered),
        len(srids_public_filtered - srids_parks),
    )
    assert (
        len(srids_parks - srids_public_filtered) <= 5
    )  # Check that all parks SRs are in public SRs

    assert (
        len(srids_public_filtered - srids_parks) <= 5
    )  # Check that (almost) all public SRs are in parks SRs; I know that a few are missing

    dfjoined = dfparks.merge(
        dfpublic_filtered, on="SRID", how="left", suffixes=["", "_y"]
    )
    return dfjoined

def add_incident_global_ID(
    df, parentcol="ServiceRequestParentGlobalID"
):  # Note: use the public data's column, "ServiceRequestParentGlobalID", not the private column, "SR ServiceRequestParentGlobalID".
    
    df.loc[:, "IncidentGlobalID"] = df.apply(
        lambda x: x[parentcol] if not type(
            x[parentcol]) == float else x["GlobalID"],
        axis=1,
    )
    return df

def drop_internal_department_reports(df):
    internal_data_sources = [
        'AMPS', 'DPR', 'DOT', 'Park Inspection Program', 'FDNY'
    ] # reports by non-public sources are also in the data; remove them.
    
    df = df[~df.SRSource.isin(internal_data_sources)]
    return df


def drop_when_reports_are_duplicted_because_of_inspections(df):
    # Some reports are inspected multiple times, and so the same report is duplicated. This removes them so the rows are unique reports (but of course not unique incidents)
    df.loc[:, "SRCreatedDate"] = pd.to_datetime(df.loc[:, "SRCreatedDate"])
    df = df.sort_values(by="SRCreatedDate")
    df = df.drop_duplicates("SRID", keep="first")
    df.loc[:, "ReportNumber"] = df.groupby("IncidentGlobalID").cumcount()
    return df

def from_original_data_pipeline(
    public_filename="data/public_Forestry_Service_Requests_downloaded_20211009.csv",
    private_filename="data/internal_DPR_data_20211203.csv",
    public_filtered_filename="data/Forestry_SRs_downloaded_202111009_filtered20211030.csv",
    joined_filename="data/private20211203_join_forestrydownloaded20211009.csv",
    processed_filename="data/fullyprocessed20211203.csv",
):
    if os.path.exists(processed_filename):
        dfjoined = pd.read_csv(processed_filename)
    else:
        if os.path.exists(joined_filename):
            dfjoined = pd.read_csv(joined_filename)
        else:
            dfparks = pd.read_csv(private_filename)
            collapsecategory = ['Remove Stump', 'Rescue/Preservation', 'Pest/Disease', 'Planting Space']
            dfparks.loc[dfparks.SRCategory.isin(collapsecategory) ,'SRCategory'] = 'Other'
            if os.path.exists(public_filtered_filename):
                dfpublic_filtered = pd.read_csv(public_filtered_filename)
            else:
                dfpublic = pd.read_csv(public_filename)
                dfpublic_filtered = filter_public_df(dfpublic)
                dfpublic_filtered.to_csv(public_filtered_filename, index=False)
            dfjoined = join_public_private(dfpublic_filtered, dfparks)

            dfjoined = drop_internal_department_reports(dfjoined)
            dfjoined = add_incident_global_ID(dfjoined)
            dfjoined = drop_when_reports_are_duplicted_because_of_inspections(
                dfjoined)
            dfjoined = dph.add_inspected_column(dfjoined)
            dfjoined.to_csv(processed_filename, index=False)

    dfjoined = dph.convert_SR_cols_to_datetime(dfjoined)
    return dfjoined