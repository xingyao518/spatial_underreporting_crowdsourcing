import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def plot_overall_extra_by_borough(dffbar, filesavelabel, column = 'overall_extra_delay', how_aggregate = 'median'):
    # Plots this by Borough
    # df['overall_extra_delay'] = df['overall_extra_delay'].fillna(10000)    
    # bar plot with Borough on x axis, overall_extra_delay on y axis
    dffbar = dffbar.sort_values(by = column, ascending = False)
    dffbar.plot(kind='barh', x = 'Borough', y = column)
    
    xlabel = {
        'median': 'Median extra delay (days)',
        'sum': 'Total extra delay (days)',
        'percent': 'Additional % overall delay relative to city average'
    }[how_aggregate]
    
    plt.xlabel(xlabel)
    plt.ylabel('')
    sns.despine()
    plt.axvline(0, -1, 5, color = 'black', linestyle = '--')
    # dffbar.plot(kind='barh')
    # remove legend
    plt.legend().remove()
    plt.savefig(f'plots/{filesavelabel}_{column}{how_aggregate}.pdf', bbox_inches='tight')
    plt.show()
    # plot_bar_and_print(dffbar, label = f'{filesavelabel}overall_extra_delay_by_borough')    

def aggregate_by_borough(df, how_aggregate = 'median', column = 'overall_extra_delay', how_normalize = 'true_delay'):
    if how_aggregate == 'median':
        dffbar = df.groupby(['Borough'])[[column]].median().reset_index()
        
    elif how_aggregate == 'sum':
        dffbar = df.groupby(['Borough'])[[column]].sum().reset_index()
        
    elif how_aggregate == 'percent':
        dffbar = df.groupby(['Borough'])[[column]].sum().reset_index()
        dffbar_to_normalize = df.groupby(['Borough'])[[how_normalize]].sum().reset_index()
        
        dffbar = pd.merge(dffbar, dffbar_to_normalize, on = 'Borough', how = 'left')
        dffbar[column] = dffbar.eval(f'({column} / {how_normalize})*100')
        
    return dffbar
        


def equity_analysis(df, filesavelabel, delay_kinds = ['overall_extra_delay', 'risk_weighted_extra_delay']
                    , statistics = ['median', 'sum', 'percent']):
    # city average delay for each kind
    df['true_delay'] = df['reporting_delay'] + df['inspection_delay'] + df['work_delay']
    df['risk_weighted_extra_delay'] = df.eval('overall_extra_delay * RiskRating')
    df['risk_weighted_true_delay'] = df.eval('true_delay * RiskRating')
    
    for col in reversed(delay_kinds):
        for how in reversed(statistics):
            print(col, how)
            dffbar = aggregate_by_borough(df, how_aggregate = how, column = col
                                          , how_normalize = 'true_delay_prediction' if col == 'overall_extra_delay' else 'risk_weighted_true_delay_prediction')
            plot_overall_extra_by_borough(dffbar, filesavelabel = f'{filesavelabel}_equity', column = col, how_aggregate = how)
    
    return df, dffbar
