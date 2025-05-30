# Install cmdstanpy: https://mc-stan.org/cmdstanpy/installation.html

import argparse
import numpy as np
import os
from cmdstanpy import cmdstan_path, CmdStanModel
import cmdstanpy
import pandas as pd
import json

from stan_modeling.prepare_stan_data import *
from settings import covariates_cont_dict
from stan_modeling import analyze_fits

from prepare_data.public_data_filtering import *

def run_model(modelname,
              savelabel='',
              treesrootdirectory='',
              iter_warmup=300,
              iter_sampling=300,
              max_incidents=-1,
              model_folder='stan_modeling/models',
              covariates_cont=covariates_cont_dict['nocens2'],
              settings=None,
              settings_name=None, show_progress=True, force_compile = False, tract11_or_block12 = 11,
              max_duration = 100, sample_size = -1):
    trees_root_directory = treesrootdirectory

    # # Load/compile stan file
    model_directory = trees_root_directory + model_folder
    model_filename = '{}.stan'.format(modelname)
    stan_file = os.path.join(model_directory, model_filename)
    savelocation_datachains = trees_root_directory + 'standata_chains'
    savelocation_modeloutput = trees_root_directory + 'stan_output'
    savefilename = '{}_{}'.format(modelname, savelabel)

    cpp_options = {
        'STAN_THREADS': True,
        'STAN_CPP_OPTIMS': True
    }

    if force_compile:
        comp= "force"
    else:
        comp = True
    model = CmdStanModel(stan_file=stan_file, cpp_options=cpp_options,
                         model_name=savefilename, compile = comp)
    print(model)

    # # Load/prepare data
    _, aggdf = public_data_pipeline_with_settings(settings={'max_duration': max_duration, 'sample_size': sample_size})

    if max_incidents > 0 and max_incidents < aggdf.shape[0]:
        aggdf = aggdf.sample(max_incidents, random_state=42)

    # get data in stan format
    data, column_names, standardization_dict = get_data_dictionary(
        aggdf, modelname, covariates_cont=covariates_cont, tract11_or_block12 = tract11_or_block12)

    parameters_to_save = {
        "modelname": modelname,
        "savelabel": savelabel,
        "treesrootdirectory": treesrootdirectory,
        "iter_warmup": iter_warmup,
        "iter_sampling": iter_sampling,
        "max_incidents": max_incidents,
        "model_folder": model_folder,
        "covariates_cont": covariates_cont,
        "settings": settings,
        "settings_name": settings_name,
        'standardization_dict': standardization_dict
    }
    column_names.update(parameters_to_save)

    # save the column names in order to be able to interpret later
    colnamesfilename = '{}/{}_colnames.txt'.format(savelocation_modeloutput, savefilename)
    datasfilename = '{}/{}_data.txt'.format(savelocation_datachains, savefilename)

    print('number incidents: ', data['N_incidents'])

    if os.path.exists(colnamesfilename) and os.path.exists(datasfilename):
        print("Skipping this stan model run since already there", colnamesfilename)
    else:
        with open(colnamesfilename, 'w') as colnamefile:
            colnamefile.write(json.dumps(column_names))

        with open(datasfilename, 'w') as datafile:
            datafile.write(json.dumps(data, cls=NumpyEncoder))

        # fit the model, save results
        fit = model.sample(data=data, iter_warmup=iter_warmup,
                        iter_sampling=iter_sampling, refresh=100, show_progress=show_progress)

        fit.save_csvfiles(dir='{}/'.format(savelocation_datachains))

        analyze_fits.analyze_fit_pipeline(
            savelocation_modeloutput, savelocation_datachains, savefilename, diagnose=True, fit_only=False, save_results=True, fit=fit)

    
    analyze_fits.save_y_and_yrep_from_saved_fit(savelocation_modeloutput, savelocation_datachains, savefilename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # add arguments to the parser
    parser.add_argument("--modelname", type = str, default = "basic_zeroinflated")
    parser.add_argument("--savelabel", type=str, default='')
    parser.add_argument("--treesrootdirectory", type=str,
                        default='/home/zl724/trees/nyc/')
    parser.add_argument("--iter_warmup", type=int, default=300)
    parser.add_argument("--iter_sampling", type=int, default=300)
    parser.add_argument("--max_incidents", type=int, default=-1)
    parser.add_argument("--model_folder", type=str,
                        default='stan_modeling/models')
    parser.add_argument("--settings_name", type=str,
                        default=None)
    parser.add_argument("--covariates_cont_name", type=str,
                        default='nocens2')
    parser.add_argument("--tract11_or_block12", type=int, default=11)
    parser.add_argument("--max_duration", type=int, default=100)
    parser.add_argument("--sample_size", type=int, default=-1)


    # parse the arguments
    args = parser.parse_args()

    for demo_var in ['loghouseholdincome', 'logdensity', 'median_value_usd', 'frac_hispanic', 'frac_white', 'frac_black', 'frac_no_hs_degree', 'frac_college_degree', 'frac_rented', 'frac_family', 'frac_poverty', 'med_age']:

        covariates_cont = ['INSP_RiskAssessment', 'INSPCondtion', 'logTPDBH']
        covariates_cont.append(demo_var)

        run_model(modelname=args.modelname,
              savelabel=args.savelabel+'_'+demo_var,
              treesrootdirectory=args.treesrootdirectory,
              iter_warmup=args.iter_warmup,
              iter_sampling=args.iter_sampling,
              max_incidents=args.max_incidents,
              model_folder=args.model_folder,
              covariates_cont=covariates_cont,
              settings_name=args.settings_name, force_compile=True, tract11_or_block12 = args.tract11_or_block12,
              max_duration = args.max_duration, sample_size = args.sample_size)
