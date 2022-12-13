import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pipeline import pipeline, pipeline_with_aggdf
from settings import *
import random


foldersave = 'aggdfs_clean'

def analyze_aggdf(aggdf, settings):
    print(aggdf.count())
    aggdf.NumberDuplicates.hist()
    plt.yscale('log')
    plt.show()
    aggdf.Duration.hist(bins = range(0, settings['max_duration']+1, 1))
    plt.yscale('log')
    plt.show()
    
def create_dfs_for_setting(rawdf, settings_name, model_exploration_ids, dfsizes = {
    'small' : 10000, 'tiny' : 1000, 'medium' : 25000, 'medbig' : 50000}):
    print(settings_name)
    aggdf = pipeline_with_aggdf(rawdf = rawdf, settings = settings_list[settings_name], settings_name = None, remove_low_categories=True)
    # analyze_aggdf(aggdf, settings_list[settings_name])

    if settings_name == 'dur100_repmedium': #save just intersection with exploration ids for an exploration dataset
        aggdf[aggdf.IncidentGlobalID.isin(model_exploration_ids)].to_csv('{}/modeldevelopment_{}.csv'.format(foldersave, settings_name), index = False)

    #remove exploration ids
    aggdf = aggdf[~aggdf.IncidentGlobalID.isin(model_exploration_ids)]

    aggdf.to_csv('{}/{}.csv'.format(foldersave, settings_name), index = False)

    for name in dfsizes:
        size=  dfsizes[name]
        aggdf.sample(n = size, random_state = 42).to_csv('{}/{}_{}.csv'.format(foldersave, name, settings_name), index = False)
    