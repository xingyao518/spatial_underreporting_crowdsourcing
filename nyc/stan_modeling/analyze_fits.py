import os
import sys
import seaborn as sns
import numpy as np
from stan_modeling.prepare_stan_data import *
import pandas as pd
import copy
import arviz.labels as azl
from cmdstanpy import cmdstan_path, CmdStanModel
import cmdstanpy
import matplotlib.pyplot as plt
import json
import arviz as az


def load_fits(savelocation, filename_template):
    fit = cmdstanpy.from_csv(
        path='{}/{}-*'.format(savelocation, filename_template))
    print(fit)
    return fit


def load_run_parameters_and_data(savelocation_outputs, savelocation_datachains, filename, load_data = True):
    with open('{}/{}_colnames.txt'.format(savelocation_outputs, filename)) as f:
        run_parameters = json.load(f)
    for key in run_parameters:
        print(key, end = ' ')
        if type(run_parameters[key]) not in [int, float, dict] and run_parameters[key] is not None:
            print(len(run_parameters[key]), run_parameters[key][0:5])
        else:
            print(run_parameters[key])
            

    if load_data:
        with open('{}/{}_data.txt'.format(savelocation_datachains, filename)) as f:
            data = json.load(f)
    else:
        data = None

    return run_parameters, data


def print_basic_fit_info(fit, diagnose=True):
    # print column names
    # colnames = fit.column_names
    # colnames_things_removed = [x.replace('[', '').replace(
    #     ']', '').replace('\d+', '') for x in colnames]
    # print(set(colnames_things_removed))

    vars = fit.stan_variables()
    for (k, v) in vars.items():
        print(k, v.shape)

    # TODO need to edit this for large data thing since this seems to crash -- can't do summary when there are 160k rows, 1 for each y_rep and log likelihood

    if diagnose:
        dd = fit.summary().reset_index()
        print('max Rhat:', dd.R_hat.max())
        if dd.R_hat.max() > 1.1:
            dd.R_hat.hist(bins=20)
            plt.show()
        print(fit.diagnose())


def replace_name(sumdf, param_name, true_names):
    to_replace = ['{}[{}]'.format(param_name, en)
                  for en in range(1, len(true_names)+1)]
    sumdf.replace(to_replace, true_names, inplace=True)


def get_fit_summary_with_pretty_cols(fit, run_parameters, name_pairs):
    sumdf = fit.summary().reset_index()
    for pair in name_pairs:
        if pair[1] in run_parameters:
            replace_name(sumdf, pair[0], run_parameters[pair[1]])
    sumdf.replace('intercept', 'Intercept', inplace=True)
    sumdf.replace('theta_zeroinflation', 'Zero Inflation fraction', inplace=True)

    return sumdf


def get_arviz_data(fit, run_parameters, name_pairs, model_data=None):
    coords = {'{}_dim'.format(pair[0]): run_parameters[pair[1]]
              for pair in name_pairs if pair[1] in run_parameters}
    dims = {pair[0]: ['{}_dim'.format(pair[0])] for pair in name_pairs}
    dataaz = az.from_cmdstanpy(
        posterior=fit,
        coords=coords,
        dims=dims,
        observed_data={"y": model_data['y']},
        posterior_predictive=['y_rep'],
        log_likelihood="log_likelihood"
    )
    var_name_map = {'intercept': 'Intercept'}
    var_name_map.update({pair[0]: '' for pair in name_pairs})

    labeller = azl.MapLabeller(
        var_name_map=var_name_map)

    return dataaz, labeller


def posterior_prediction_checks(dataaz, data_pairs={"y": "y_rep"}, maxhistval=20, save_results=False, save_template=None):
    az.plot_ppc(dataaz, data_pairs=data_pairs,
                alpha=0.03, figsize=(12, 6), textsize=14)
    plt.yscale('log')
    if save_results and save_template is not None:
        plt.savefig('{}_ppc.png'.format(save_template), bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    posty_means = dataaz.posterior_predictive.y_rep.mean(axis=1).mean(axis=0)
    posty_flat = np.ndarray.flatten(
        np.array(dataaz.posterior_predictive.y_rep))

    plt.hist(dataaz.observed_data.y, bins=range(0, maxhistval),
             alpha=.5, cumulative=True, density=True, label='observed')

    plt.hist(posty_flat, bins=range(0, maxhistval),
             alpha=.5, cumulative=True, density=True, label='posterior_sample')
    plt.yscale('log')
    plt.legend()
    if save_results and save_template is not None:
        plt.savefig('{}_histogramcdfs.png'.format(
            save_template), bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    _ = plt.hist(dataaz.observed_data.y, bins=range(
        0, maxhistval), alpha=.5, density=True, label='observed')
    _ = plt.hist(posty_flat, bins=range(0, maxhistval), alpha=.5,
                 density=True, label='posterior_sample')
    plt.yscale('log')
    plt.legend()
    if save_results and save_template is not None:
        plt.savefig('{}_histograms.png'.format(
            save_template), bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    az.plot_bpv(dataaz, data_pairs=data_pairs, kind="p_value")
    if save_results and save_template is not None:
        plt.savefig('{}_bpv_pval.png'.format(
            save_template), bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    az.plot_bpv(dataaz, data_pairs=data_pairs, kind="t_stat",
                t_stat=lambda x: np.percentile(x, q=50, axis=-1))
    if save_results and save_template is not None:
        plt.savefig('{}_bpv_tstat.png'.format(
            save_template), bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    _ = sns.jointplot(x=dataaz.observed_data.y, y=posty_means, kind="reg")
    if save_results and save_template is not None:
        plt.savefig('{}_jointreg.png'.format(
            save_template), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def more_convergence_checks(dataaz, arviz_to_plot, labeller, save_results=False, save_template=None):
    _ = az.plot_posterior(dataaz, var_names=arviz_to_plot, labeller=labeller)
    if save_results and save_template is not None:
        plt.savefig('{}_posterior.png'.format(
            save_template), bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    _ = az.plot_rank(dataaz, var_names=arviz_to_plot, labeller=labeller)
    if save_results and save_template is not None:
        plt.savefig('{}_rankcheck.png'.format(
            save_template), bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    _ = az.plot_pair(dataaz,
                     var_names=["beta"],
                     kind='kde',
                     labeller=labeller,
                     divergences=True,
                     textsize=18)
    if save_results and save_template is not None:
        plt.savefig('{}_pairplot.png'.format(
            save_template), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def analyze_fit_pipeline(savelocation_outputs, savelocation_datachains, filename_template, diagnose=True, name_pairs=[('beta', 'X'), ('beta_borough', 'Borough'), ('beta_category', 'Category'), ('beta_tract', 'Tract'), ('beta_CommunityBoard', 'CB'), ('beta_month', 'Month')], ignore_vals_for_pretty_sumdf=['_raw', '_total', 'lp__', 'y_rep', 'log_likelihood'], data_pairs={"y": "y_rep"}, maxhistval=20, arviz_to_plot=['beta', 'beta_borough', 'beta_category'], fit_only=False, save_results=False, just_loading=True, also_load_fit=False, fit=None, load_data = True
                         ):

    original_stdout = sys.stdout  # Save a reference to the original standard output

    # if save_results == True:
    #     fileout = open(
    #         '{}/resultsanalyze_{}.txt'.format(savelocation_outputs, filename_template), 'w')
    #     sys.stdout = fileout

    if fit is None:
        if also_load_fit:
            fit = load_fits(savelocation_datachains, filename_template)
        if fit_only:
            return fit

    run_parameters, model_data = load_run_parameters_and_data(
        savelocation_outputs, savelocation_datachains, filename_template, load_data=load_data or also_load_fit)

    # print(model_data)
    save_template = '{}/{}'.format(savelocation_outputs, filename_template)

    if diagnose:
        sumdffilename = '{}/sumdf_{}.csv'.format(savelocation_outputs,
                                                 filename_template)
        prettydffilename = '{}/prettysumdf_{}.csv'.format(savelocation_outputs,
                                                          filename_template)

        if not os.path.isfile(sumdffilename):
            sumdf = get_fit_summary_with_pretty_cols(
                fit, run_parameters, name_pairs=name_pairs)

            prettysumdf = copy.copy(sumdf)
            prettysumdf = prettysumdf.rename(columns={'index':'name'})
            for val in ignore_vals_for_pretty_sumdf:
                prettysumdf = prettysumdf[~prettysumdf.name.str.contains(val)]
            prettysumdf = pd.concat(
                [prettysumdf.query('name=="Intercept"'), prettysumdf.query('name=="Zero Inflation fraction"'), prettysumdf.query('(name != "Intercept") and (name != "Zero Inflation fraction")')]).reset_index(drop = True)

            if save_results:
                sumdf.to_csv(sumdffilename, index=False)
                prettysumdf.to_csv(prettydffilename, index=False)
        else:
            print('loading sumdf from file')
            sumdf = pd.read_csv(sumdffilename)
            prettysumdf = pd.read_csv(prettydffilename)
            prettysumdf.replace('theta_zeroinflation', 'Zero Inflation fraction', inplace=True)
            prettysumdf = pd.concat(
                [prettysumdf.query('name=="Intercept"'), prettysumdf.query('name=="Zero Inflation fraction"'), prettysumdf.query('(name != "Intercept") and (name != "Zero Inflation fraction")')]).reset_index(drop = True)

    else:
        sumdf = None
        prettysumdf = None

    if also_load_fit:
        dataaz, labeller = get_arviz_data(
            fit, run_parameters, name_pairs, model_data)
    else:
        dataaz = None
        labeller = None
    if not just_loading and also_load_fit:
        if not any(prettysumdf.name.str.contains('borough')):
            arviz_to_plot = [x for x in arviz_to_plot if x != 'beta_borough']
        (ax, ) = az.plot_forest(dataaz, var_names=arviz_to_plot, labeller=labeller)
        ax.axvline(0, color='k')
        if save_results and save_template is not None:
            plt.savefig('{}_forest.png'.format(
                save_template), bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        try:
            posterior_prediction_checks(
                dataaz, data_pairs=data_pairs, maxhistval=maxhistval, save_results=save_results, save_template=save_template)
        except Exception as e:
            print('Error in postior_prediction_checks: {}'.format(e))

        try:
            more_convergence_checks(dataaz, arviz_to_plot, labeller,
                                    save_results=save_results, save_template=save_template)
        except Exception as e:
            print('Error in convergence check: {}'.format(e))

        print_basic_fit_info(fit, diagnose=diagnose)
        sys.stdout = original_stdout  # Reset the standard output to its original value
        
    return fit, run_parameters, model_data, sumdf, prettysumdf, dataaz, labeller

def save_y_and_yrep_from_saved_fit(savelocation_modeloutput, savelocation_datachains, filename_template):
    if not os.path.exists('{}/{}_y.txt'.format(savelocation_modeloutput, filename_template)):
        print('saving y data from fits')
        fit = load_fits(savelocation_datachains, filename_template)
        run_parameters, model_data = load_run_parameters_and_data(
            savelocation_modeloutput, savelocation_datachains, filename_template, load_data=True)

        yrep = fit.stan_variable('y_rep')
        y = model_data['y']
        yrep_means = yrep.mean(axis = 0)
        yrep_last10samples = yrep[-10:, :]

    
        with open('{}/{}_y.txt'.format(savelocation_modeloutput, filename_template), 'w') as f:
            np.savetxt(f, y)
        with open('{}/{}_yrep_means.txt'.format(savelocation_modeloutput, filename_template), 'w') as f:
            np.savetxt(f, yrep_means)
        with open('{}/{}_yrep_last10samples.txt'.format(savelocation_modeloutput, filename_template), 'w') as f:
            np.savetxt(f, yrep_last10samples)


def load_y_and_yrep_from_saved_txt(savelocation, filename_template):
    # yrep = np.loadtxt('{}/{}_yrep.txt'.format(savelocation, filename_template))
    y = np.loadtxt('{}/{}_y.txt'.format(savelocation, filename_template))
    yrep_means = np.loadtxt('{}/{}_yrep_means.txt'.format(savelocation, filename_template))
    yrep_last10samples = np.loadtxt('{}/{}_yrep_last10samples.txt'.format(savelocation, filename_template))
    return y, yrep_means, yrep_last10samples

def load_ys_and_print_correlation(savelocation_modeloutput, filename_template):
    import scipy.stats as stats

    with open('{}/{}_y.txt'.format(savelocation_modeloutput, filename_template), 'r') as f:
        ys = np.loadtxt(f)
    with open('{}/{}_yrep_means.txt'.format(savelocation_modeloutput, filename_template), 'r') as f:
        yrep_means = np.loadtxt(f)
    # print(ys, yrep_means)
    print(stats.pearsonr(ys, yrep_means))