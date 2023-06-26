import json
import numpy as np
from patsy import dmatrices, dmatrix
import patsy
from patsy import *
import pickle
import scipy

def get_edge_lists_for_tract_adjacency(aggdftractorder, adjacency_file = 'data/tract_adjacency_connected.npz', tract_adjacency_order_file = 'data/tract_order_adjacency'):
    A = scipy.sparse.load_npz(adjacency_file)
    tractorderadjacency_list = pickle.load(open(tract_adjacency_order_file, 'rb') )

    print(aggdftractorder[0:5])
    print(tractorderadjacency_list[0:5])

    Adock = A.todok().keys()
    Adock = list(sorted(set([tuple(sorted(x)) for x in Adock])))

    node1 = []
    node2 = []
    
    orig_mapping_to_node_lists = {} #covert adjacency matrix 0 indexed thing to node lists 1 index thing, but aligning the actual tract numbers for each
    for en, (n1, n2) in enumerate(Adock):
        if n1 not in orig_mapping_to_node_lists:
            try:
                orig_mapping_to_node_lists[n1] = aggdftractorder.index(tractorderadjacency_list[n1]) + 1
            except ValueError:
                orig_mapping_to_node_lists[n1] = -1 #not in aggdftractorder
        if n2 not in orig_mapping_to_node_lists:
            try:
                orig_mapping_to_node_lists[n2] = aggdftractorder.index(tractorderadjacency_list[n2]) + 1
            except ValueError:
                orig_mapping_to_node_lists[n2] = -1 #not in aggdftractorder
        if (orig_mapping_to_node_lists[n1] == -1) or (orig_mapping_to_node_lists[n2] == -1):
            continue
        node1.append(orig_mapping_to_node_lists[n1])
        node2.append(orig_mapping_to_node_lists[n2])
    print(len(node1), len(set.union(set(node1),set(node2))), len(aggdftractorder))
    return node1, node2
    
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
# to save this in json: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_data_dictionary(aggdf, modelname, maxdatapoints=None, covariates_cont=None, tract11_or_block12 = 11):
    if "observables" in modelname:
        return get_data_dictionary_observables(aggdf, modelname, maxdatapoints=maxdatapoints, covariates_cont=covariates_cont, tract11_or_block12 = tract11_or_block12)

    column_names = {}  # used for plotting (saving the order of the columns)
    covariates = []  # ['Category']
    othercov = covariates_cont
    covariates += othercov
    
    ## sort the months for easier sequencing
    if "temporal" in modelname:
        aggdf.SRCreatedDate = aggdf.SRCreatedDate.astype('datetime64[ns]').dt.to_period('M')
        aggdf = aggdf.sort_values(by = 'SRCreatedDate').reset_index()
        aggdf.SRCreatedDate = aggdf.SRCreatedDate.astype('str')

    if maxdatapoints is not None:
        aggdf = aggdf.iloc[0:maxdatapoints, :]

    colstodo = ['NumberDuplicates', 'Duration'] + ['Borough'] + covariates
    aggdfloc = aggdf.dropna(subset=colstodo)

    # standardize myself instead of using patsy so that I can save the standardization parameters so that I can undo it when post-stratifying
    standardization_dict = {}
    for col in covariates:
        try:
            standardization_dict[col] = {
                'mean': aggdfloc[col].dropna().mean(), 'std': aggdfloc[col].dropna().std()}
            aggdfloc.loc[:, col] = (
                aggdfloc[col] - standardization_dict[col]['mean']) / standardization_dict[col]['std']
        except Exception as e:
            print(
                'Error standardizing {} (most likely fine, not a float/int)'.format(col))

    ydm, Xdm = dmatrices(
        'NumberDuplicates ~ 1 + {}'.format(' + '.join(covariates)), data=aggdfloc)  # doing 1+ and then removing the intercept (have it separately) so that categorical variables don't have too many DoF

    for cat in covariates:
        print(cat, list(aggdfloc[cat].unique())[0:5],
              standardization_dict.get(cat, ''))

    X = np.asarray(Xdm)[:, 1:]  # remove intercept
    y = [int(yy[0]) for yy in np.asarray(ydm)]

    duration = (aggdfloc.Duration).values

    _, X_borough_dm = dmatrices(
        'NumberDuplicates ~ 0 + Borough', data=aggdfloc)
    X_borough = np.asarray(X_borough_dm)

    _, X_category_dm = dmatrices(
        'NumberDuplicates ~ 0 + Category', data=aggdfloc)
    X_category = np.asarray(X_category_dm)

    if "temporal" in modelname:
        _, X_month_dm = dmatrices(
            'NumberDuplicates ~ 0 + SRCreatedDate', data=aggdfloc)
        X_month = np.asarray(X_month_dm)

        column_names['Month'] = X_month_dm.design_info.column_names

    column_names['X'] = Xdm.design_info.column_names[1:]
    column_names['Borough'] = X_borough_dm.design_info.column_names
    column_names['Category'] = X_category_dm.design_info.column_names
    column_names['Category_Zeroinflation']= ['{}_inflation'.format(X) for X in X_category_dm.design_info.column_names]

    ones = np.ones(len(y))
    data = {"N_incidents": len(y), "y": y, "duration": duration, "X": X, "X_borough": X_borough, "X_category": X_category,
            "N_category": np.shape(X_category)[1], "ones": ones, "covariate_matrix_width": np.shape(X)[1]}
    

    if "interaction" in modelname:
        _, X_borough_category_dm = dmatrices(
            'NumberDuplicates ~ 1 + Borough + Category + Borough:Category', data=aggdfloc)
        X_borough_category = np.asarray(X_borough_category_dm)[:, -16:]  # remove intercept and the single terms
        data['X_borough_category'] = X_borough_category
        data['N_borough_category'] = np.shape(X_borough_category)[1]
        column_names['Borough_Category'] = X_borough_category_dm.design_info.column_names[-16:]
    
    if "riskbin" in modelname:
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

        # Apply the function to each row of the DataFrame
        aggdfloc['Risk_coded'] = aggdfloc['INSP_RiskAssessment'].apply(map_priority)

        _, X_risk_dm = dmatrices(
            'NumberDuplicates ~ 0 + Risk_coded', data=aggdfloc)
        X_risk = np.asarray(X_risk_dm)
        data['X_risk'] = X_risk
        data['N_risk'] = np.shape(X_risk)[1]
        column_names['Risk'] = X_risk_dm.design_info.column_names

        _, X_borough_riskbin_dm = dmatrices(
            'NumberDuplicates ~ 1 + Borough + Risk_coded + Borough:Risk_coded', data=aggdfloc)
        X_borough_riskbin = np.asarray(X_borough_riskbin_dm)[:, -4*(data['N_risk'] -1):]  # remove intercept and the single terms
        data['X_borough_riskbin'] = X_borough_riskbin
        data['N_borough_riskbin'] = np.shape(X_borough_riskbin)[1]
        column_names['Borough_Riskbin'] = X_borough_riskbin_dm.design_info.column_names[-4*(data['N_risk'] -1):]
 

    if "temporal" in modelname:
        data['X_month'] = X_month
        data['N_month'] = np.shape(X_month)[1]

    if modelname in ['basic', 'basic_zeroinflated', 'basic_zeroinflated_noborough', \
                     'basic_zeroinflated_temporal', 'basic_zeroinflated_temporal_year', 'basic_zeroinflated_bycategory','basic_zeroinflated_gen_delay',\
                     'basic_zeroinflated_borough_category_interaction', "basic_zeroinflated_only_interaction",
                     "basic_zeroinflated_riskbin"]:
        if modelname == 'basic_zeroinflated_noborough':
            del data['X_borough']
        
        return data, column_names, standardization_dict

    aggdfloc.loc[:, 'census_tract'] = aggdfloc.loc[:,'census_tract'].apply(lambda x: str(int(x))[0:tract11_or_block12])
    _, X_tract_dm = dmatrices(
        'NumberDuplicates ~ 0 + census_tract', data=aggdfloc)
    X_tract = np.asarray(X_tract_dm)

    column_names['Tract'] = X_tract_dm.design_info.column_names
    column_names['length_of_census_tract_column'] = tract11_or_block12

    data.update({"X_tract": X_tract, "N_tract": np.shape(X_tract)[1]})

    if 'tract_adjacency' in modelname:
        for coldel in ['X_borough']:
            del data[coldel]

        tract_order_clean = [x.replace('census_tract[', '').replace(']', '') for x in column_names['Tract']]
        node1, node2 = get_edge_lists_for_tract_adjacency(tract_order_clean) 
        n_edges = len(node1)
        data.update({"node1": node1, "node2": node2, "N_edges": n_edges})
        return data, column_names, standardization_dict

    assert False


def get_data_dictionary_observables(aggdf, modelname, maxdatapoints=None, covariates_cont=None, tract11_or_block12 = 11):
    column_names = {}  
    
    if maxdatapoints is not None:
        aggdf = aggdf.iloc[0:maxdatapoints, :]

    colstodo = ['NumberDuplicates', 'Duration'] + ['Borough'] + ['Category']
    aggdfloc = aggdf.dropna(subset=colstodo)

    ydm, _ = dmatrices(
        'NumberDuplicates ~ 1', data=aggdfloc) 
    y = [int(yy[0]) for yy in np.asarray(ydm)]

    duration = (aggdfloc.Duration).values

    _, X_borough_dm = dmatrices(
        'NumberDuplicates ~ 0 + Borough', data=aggdfloc)
    X_borough = np.asarray(X_borough_dm)

    _, X_category_dm = dmatrices(
        'NumberDuplicates ~ 0 + Category', data=aggdfloc)
    X_category = np.asarray(X_category_dm)

    column_names['Borough'] = X_borough_dm.design_info.column_names
    column_names['Category'] = X_category_dm.design_info.column_names

    ones = np.ones(len(y))
    data = {"N_incidents": len(y), "y": y, "duration": duration, "X_borough": X_borough, "X_category": X_category,
            "N_category": np.shape(X_category)[1], "ones": ones}
    
    _, X_borough_category_dm = dmatrices(
        'NumberDuplicates ~ 1 + Borough + Category + Borough:Category', data=aggdfloc)
    X_borough_category = np.asarray(X_borough_category_dm)[:, -16:]  # remove intercept and the single terms
    data['X_borough_category'] = X_borough_category
    data['N_borough_category'] = np.shape(X_borough_category)[1]
    column_names['Borough_Category'] = X_borough_category_dm.design_info.column_names[-16:]
    
    return data, column_names, 0

