import pandas as pd
import os

def save_pretty_df_for_coefficients(prettydf, modelname, dir_savefiles, filename_template, additional_caption = ''):
    
    prettydf.loc[:, 'name'] = prettydf.loc[:, 'name'].replace('logTPDBH', 'Log(Tree Diameter at Breast Height)')
    prettydf = prettydf[~prettydf.name.str.contains('offset') & ~prettydf.name.str.contains('census') & ~prettydf.name.str.contains('CommunityBoard')]

    caption = 'Regression coefficients for {} with incident-level covariates and Borough fixed effects{}.'.format(modelname, additional_caption)

    cols_for_latex = ['name', 'Mean', 'StdDev', '5%', '50%', '95%', 'R_hat']
    
    filename = '{}_latexsummary.tex'.format(filename_template)
    prettydf[cols_for_latex].to_latex(os.path.join(dir_savefiles, filename), index = False, caption = caption)
    return filename


def generate_summary_table_from_raw_df(df, groupby = None):
    
    first_report_and_inspection_times = df.groupby("IncidentGlobalID"
                                                    ).agg(**{
                                                            'first_report_datetime': pd.NamedAgg('SRCreatedDate', "min"),
                                                            'first_inspection_datetime': pd.NamedAgg('InspectionDate', "min"),
                                                            'workorder_datetime': pd.NamedAgg('ActualFinishDate', "min"),

                                                            }).reset_index()
    first_report_and_inspection_times.loc[:, 'days_after_first_report'] = first_report_and_inspection_times.first_report_datetime + \
        pd.Timedelta(100, unit='D')
    first_report_and_inspection_times.loc[:, 'death_time'] = first_report_and_inspection_times['first_inspection_datetime']

    first_report_and_inspection_times.loc[:, 'Duration'] = (
        first_report_and_inspection_times.death_time - first_report_and_inspection_times.first_report_datetime).dt.total_seconds() / (60 * 60 * 24)  # Duration in days

    df = pd.merge(df, first_report_and_inspection_times[[
                'IncidentGlobalID', 'Duration']], on="IncidentGlobalID")        
    
    grouped = df.groupby(groupby).agg({'SRID': 'count', 'Inspected': 'sum'}).sort_values(ascending=False, by = 'SRID').rename(columns = {'SRID': 'Requests', 'Inspected': 'Inspected Requests'})
    grouped.loc[:, 'Fraction Inspected'] = grouped.eval('`Inspected Requests` / Requests').round(2)
    
    
    unique_incidents = df.query('Inspected == 1').groupby(groupby)['IncidentGlobalID'].nunique().sort_values(ascending=False).reset_index().rename(columns = {'IncidentGlobalID': 'Unique Incidents'})
    inspection_delays = df.query('(Inspected == 1)').drop_duplicates(subset = ['IncidentGlobalID']).groupby(groupby)['Duration'].median().reset_index().rename(columns = {'Duration': 'Median Time to Inspection (Days)'})
    
    inspection_delays.loc[:, 'Median Time to Inspection (Days)'] = inspection_delays['Median Time to Inspection (Days)'].round(2)
    
    grouped = pd.merge(grouped, unique_incidents, on = groupby)
    grouped.loc[:,'Avg Reports/Incident'] = grouped.eval('`Inspected Requests` / `Unique Incidents`').round(2)
    grouped = grouped.merge(inspection_delays, on = groupby).rename(columns = {groupby: groupby.replace('SR', '')})
    print(grouped)
    print(grouped.to_latex(index = False))
    return grouped