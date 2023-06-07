import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function to apply the mapping to each row
def map_priority(risk):
    if np.isnan(risk):
        return 'Unknown'
    if risk <3: 
        return 'E'
    if risk < 9:
        return 'D'
    if risk < 10:
        return 'C'
    if risk < 11:
        return 'B'
    return 'A'

def plot_bar_and_print(dffbar, label = ''):

    dffbar['summ'] = dffbar.eval('reporting_delay + inspection_delay + work_delay')
    dffbar = dffbar.sort_values(by = 'summ', ascending = False)
    ax = dffbar[['reporting_delay', 'inspection_delay',  'work_delay']
        ].plot(kind='barh', stacked=True, color=['red', 'skyblue', 'green'])
    
    # for container in ax.containers:
        # ax.bar_label(container, fmt='%.1f', label_type='center', fontsize = 7)
    
    # Add labels above the bars
    legthssofar = {}
    for barconts in ax.containers:
        # print(len(barconts))
        for en, bar in enumerate(barconts):
            widthsofar = legthssofar.get(en, 0)
            width = bar.get_width()
            startwidth = max(widthsofar + width/2 - .5, widthsofar + .25)
            if widthsofar == 0:
                startwidth = widthsofar + width/2 - .5
            ax.text(startwidth, bar.get_y() + bar.get_height()+.1, '{:.1f}'.format(width) , ha='left', va='center', fontsize = 9)
            legthssofar[en] = widthsofar + width

        
    sns.despine()
    plt.legend(frameon = False, labels = ['Median Est. Reporting delay', 'Median Inspection delay', 'Median Work order delay'])
    plt.xlabel('Days')
    # plt.ylabel('Borough')
    plt.ylabel('')

    #replace "/" in label so can print to file
    label = label.replace('/', '_')
    plt.savefig(f'plots/{label}.pdf', bbox_inches='tight')
    plt.show()
    
def plot_bar_by_type(df, typecol = 'Category', othergroupby = 'Borough', do_work_delay = True, impute_missing_work_order = True, label = ''):
    dffloc = df.copy()
    
    if impute_missing_work_order:
        dffloc['work_delay'] = dffloc['work_delay'].fillna(10000)
    
    
    #########
    print('Overall split by typecol but not other group')
    dffbar = dffloc.groupby([typecol])[['reporting_delay', 'inspection_delay',  'work_delay']].median()
    plot_bar_and_print(dffbar, label = f'{label}Overall_{typecol}')
    
    ##############
    for cat in dffloc[typecol].unique():
        print(cat)
        dffloccat =  dffloc.query(f'{typecol} == "{cat}"').copy()
        
        dffbar = dffloccat.groupby(othergroupby)[['reporting_delay', 'inspection_delay',  'work_delay']].median()
        plot_bar_and_print(dffbar, label = f'{label}{cat}_{othergroupby}')
        