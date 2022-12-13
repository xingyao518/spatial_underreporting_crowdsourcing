import copy

settings_list = {
    'dur30_repmostconservative': {'max_duration': 30, 'repeat_caller_conservativeness': 'most_conservative'},
    'dur100_repmostconservative': {'max_duration': 100, 'repeat_caller_conservativeness': 'most_conservative'},
    'dur200_repmostconservative': {'max_duration': 200, 'repeat_caller_conservativeness': 'most_conservative'},
    'dur30_repmedium': {'max_duration': 30, 'repeat_caller_conservativeness': 'medium'},
    'dur100_repmedium': {'max_duration': 100, 'repeat_caller_conservativeness': 'medium'},
    'dur200_repmedium': {'max_duration': 200, 'repeat_caller_conservativeness': 'medium'},
    'dur30_repallcalls': {'max_duration': 30, 'repeat_caller_conservativeness': 'use_all_calls'},
    'dur100_repallcalls': {'max_duration': 100, 'repeat_caller_conservativeness': 'use_all_calls'},
    'dur200_repallcalls': {'max_duration': 200, 'repeat_caller_conservativeness': 'use_all_calls'},
}

# also have the other datasets in with "small" and "tiny" prepended
settings_list_small = {}
for k, v in settings_list.items():
    vnew = copy.copy(v)
    vnew.update({'incident_limit': 10000})
    settings_list_small['{}_{}'.format('small', k)] = vnew

settings_list_tiny = {}
for k, v in settings_list.items():
    vnew = copy.copy(v)
    vnew.update({'incident_limit': 1000})
    settings_list_tiny['{}_{}'.format('tiny', k)] = vnew

settings_default_name = 'dur30_repmedium'
aggdf_folder_default = 'aggdfs_clean'

settings_lists_dict = {
    'settings_name_list_allmain': ['dur30_repmedium', 'dur30_repmostconservative', 'dur100_repmostconservative', 'dur100_repmedium', 'dur30_repallcalls', 'dur100_repallcalls'],
    'settings_name_list_default': ['small_dur100_repmedium', 'dur100_repmedium'], #, 'dur30_repmedium'
    'settings_name_list_smalldata': list(settings_list_small.keys()),
    'settings_name_list_tinydata': list(settings_list_tiny.keys()),
    'settings_name_list_allbig': list(settings_list.keys()),
}

covariates_cont_dict = {
    'nocens2': ['INSP_RiskAssessment', 'INSPCondtion', 'logTPDBH'],
    'simulated': ['INSPCondtion', 'TPStructure']
}

census_variables = [ 'med_age',
       'frac_hispanic',
       'frac_white', 'frac_black'
     , 'frac_noHSGrad', 'frac_collegegrad', 'frac_poverty',
        'frac_renter', 'frac_single_unit',
       'loghouseholdincome', 'logdensity']

for cencol in census_variables:
    covariates_cont_dict['singlecens2_{}'.format(cencol)] = covariates_cont_dict['nocens2'] +  [cencol]


pretty_col_names = {
    'med_age': 'Median Age', 
    'frac_hispanic': 'Fraction Hispanic', 
    'frac_white': 'Fraction white', 
    'frac_black': 'Fraction Black', 
    'frac_noHSGrad': 'Fraction noHSGrad', 
    'frac_collegegrad': 'Fraction college grad', 
    'frac_poverty': 'Fraction poverty', 
    'frac_renter': 'Fraction renter', 
    'frac_single_unit': 'Fraction single unit', 
    'loghouseholdincome': 'Log(Avg income)', 
    'logdensity': 'Log(Density)', 
}

pretty_model_names = {
    'basic': 'Standard Poisson regression',
    'basic_zeroinflated': 'Zero-inflated Poisson regression',
}

pretty_settings_names = {
    'dur30_repmedium': 'Max Duration 30 days, Default repeat caller removal',
    'dur30_repmostconservative': 'Max Duration 30 days, Remove all repeat callers and missing caller information',
    'dur100_repmostconservative': 'Max Duration 100 days, Remove all repeat callers and missing caller information',
    'dur100_repmedium': 'Max Duration 100 days, Default repeat caller removal',
    'dur200_repmostconservative': 'Max Duration 200 days, Remove all repeat callers and missing caller information',
    'dur200_repmedium': 'Max Duration 200 days, Default repeat caller removal',
}
