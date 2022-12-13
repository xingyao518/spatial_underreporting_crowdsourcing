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
    column_names = {}  # used for plotting (saving the order of the columns)
    covariates = []  # ['Category']
    othercov = covariates_cont
    covariates += othercov
    
    ## sort the months for easier sequencing
    if modelname=='basic_zeroinflated_temporal':
        aggdf.SRCreatedDate = aggdf.SRCreatedDate.dt.to_period('M')
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

    _, X_month_dm = dmatrices(
        'NumberDuplicates ~ 0 + SRCreatedDate', data=aggdfloc)
    X_month = np.asarray(X_month_dm)

    column_names['Month'] = X_month_dm.design_info.column_names

    column_names['X'] = Xdm.design_info.column_names[1:]
    column_names['Borough'] = X_borough_dm.design_info.column_names
    column_names['Category'] = X_category_dm.design_info.column_names

    ones = np.ones(len(y))
    data = {"N_incidents": len(y), "y": y, "duration": duration, "X": X, "X_borough": X_borough, "X_category": X_category,
            "N_category": np.shape(X_category)[1], "ones": ones, "covariate_matrix_width": np.shape(X)[1], "X_month": X_month, "N_month": np.shape(X_month)[1]}
 

    if modelname in ['basic', 'basic_zeroinflated', 'basic_zeroinflated_noborough', 'basic_zeroinflated_temporal']:
        if modelname == 'basic_zeroinflated_noborough':
            del data['X_borough']
        
        return data, column_names, standardization_dict

    aggdfloc.loc[:, 'census_tract'] = aggdfloc.loc[:,
                                                   'census_tract'].apply(lambda x: str(int(x))[0:tract11_or_block12])
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
