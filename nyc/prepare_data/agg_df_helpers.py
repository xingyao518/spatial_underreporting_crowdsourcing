import pandas as pd
import numpy as np

"""
This file contains helper functions for creating an aggregate dataframe to use for the rate estimation.
"""

def add_inspected_column(df):
    df.loc[:, "Inspected"] = 1 - df.InspectionDate.isna()
    inspected_df = df.groupby("IncidentGlobalID")["Inspected"].sum()
    inspectedid = list(inspected_df[inspected_df >= 1].index)
    df.loc[df.IncidentGlobalID.isin(inspectedid), 'Inspected'] = 1
    return df

def filter_inspected(df):
    """
    Filter out the rows that are not inspected.

    Note: if eventually we do our own duplicate detection, we might not want to filter out the uninspected rows, as we will have duplicates for uninspected as well.
    """
    df.loc[:, "Inspected"] = 1 - df.InspectionDate.isna()
    inspected_df = df.groupby("IncidentGlobalID")["Inspected"].sum()
    inspectedid = list(inspected_df[inspected_df >= 1].index)
    return df[df.IncidentGlobalID.isin(inspectedid)]


def convert_SR_cols_to_datetime(df):
    """
    Convert the various needed dates to datetime format
    """
    df.loc[:, "SRCreatedDate"] = pd.to_datetime(df.SRCreatedDate)
    df.loc[:, "SRClosedDate"] = pd.to_datetime(df.SRClosedDate)
    df.loc[:, "InspectionDate"] = pd.to_datetime(df.InspectionDate)
    df.loc[:, 'ActualFinishDate'] = pd.to_datetime(df.ActualFinishDate)
    return df


def filter_out_negative_reporting_periods(df):
    reportingperiod = df.groupby("IncidentGlobalID")[
        "ReportingPeriod"].agg(np.max)
    df = df[df.IncidentGlobalID.isin(
        reportingperiod[reportingperiod > 0].index)]
    return df
