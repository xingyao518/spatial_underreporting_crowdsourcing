from cmdstanpy import cmdstan_path, CmdStanModel
import cmdstanpy
import os
from stan_model import analyze_fits

from stan_model.prepare_stan_data import *
from chicago_data_filtering import *

import argparse


def run_model(modelname, 
        trees_root_directory = './',
        # trees_root_directory = '',
        iter_warmup = 300,
        iter_sampling = 300,
        max_incidents = -1,
        model_folder = 'stan_model/',
        covariates_cont = None,
        settings = None, force_compile = False, save_label = '', settings_name = '', show_progress = True,
        tract11_or_block12 = 12):

    ## load/compile stan file
        model_directory = trees_root_directory + model_folder
        model_filename = '{}.stan'.format(modelname)
        stan_file = os.path.join(model_directory, model_filename)
        savelocation_datachains = trees_root_directory + 'standata_chains'
        savelocation_modeloutput = trees_root_directory + 'stan_output'
        savefilename = '{}_{}'.format(modelname, save_label)

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
    ## load and prepare data
        if settings == None:
            _, _, aggdf = pipeline_with_settings(default_settings)
        else:
            _, _, aggdf = pipeline_with_settings(settings)

        if max_incidents>0 and max_incidents<aggdf.shape[0]:
            aggdf = aggdf.sample(max_incidents, random_state=1)
        
        data, column_names, standardization_dict = get_data_dictionary(aggdf, covariates_cont = covariates_cont)

        parameters_to_save = {
        "modelname": modelname,
        "savelabel": save_label,
        "treesrootdirectory": trees_root_directory,
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

        # if os.path.exists(colnamesfilename) and os.path.exists(datasfilename):
        if False:
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

    parser.add_argument("--model_name", type = str, default = 'chicago_5smooth')
    parser.add_argument("--iter_warmup", type=int, default=300)
    parser.add_argument("--iter_sampling", type=int, default=300)
    parser.add_argument("--max_incidents", type=int, default=-1)
    parser.add_argument("--save_label", type=str, default=os.times().system)

    args = parser.parse_args()

    run_model(modelname = args.model_name, iter_warmup=args.iter_warmup, iter_sampling=args.iter_sampling, max_incidents=args.max_incidents, save_label=args.save_label, settings_name='', show_progress=True)