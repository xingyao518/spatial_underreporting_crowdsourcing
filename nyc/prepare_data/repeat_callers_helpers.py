import pandas as pd
import numpy as np


#Parks department just hashed each number/name/email, without giving us which hashes belong to empty strings/nulls/etc. So we need to do a bit of a hack to get the right number of unique callers, without being too conservative by declaring repeat empty strings as duplicates. Hypothesis that if the same string appears in multiple of phone/email/name, then that means it's not a "true" value since a valid phone number is not a valid email or name.

#Overlap set is the set of hashes that appear in multiple columns, and so are not "true" values. 
def get_overlap_set(df, cols=['SRPhone', 'SREmail', 'SRFirstName', 'SRLastName']):
    print('Getting overlap set')
    overlapset = []
    uniques = {col: set(df[col].unique()) for col in cols}
    for en, col in enumerate(cols):
        for col2 in cols[en+1:]:
            # skip if first and last name, since human first names can also be last names
            if col == 'SRFirstName' and col2 == 'SRLastName':
                continue
            sett = uniques[col] & uniques[col2]
            overlapset.extend(sett)
    overlapset = set(overlapset)
    return overlapset

overlapset = None

def most_conservative_comparative_function(row1, row2):
    # if any of Phone, Email match, or First and Last name match, then not unique
    # not handling the fact that many of the hashes represent missing data, and so if 2 people have missing data, they'll be regarded as a match

    if (row1.SRPhone == row2.SRPhone) or (row1.SREmail == row2.SREmail) or ((row1.SRFirstName == row2.SRFirstName) and (row1.SRLastName == row2.SRLastName)):
        return True
    return False

def medium_comparative_function(row1, row2):
    # There is a set of phones/emails/names that we do not regard as same person because those represent missing values. Otherwise, we use the same rule as conservative: match phone, email, or full name.

    if (row1.SRPhone == row2.SRPhone and (row1.SRPhone not in overlapset)) or (row1.SREmail == row2.SREmail and (row1.SREmail not in overlapset)) or ((row1.SRFirstName == row2.SRFirstName and (row1.SRFirstName not in overlapset)) and (row1.SRLastName == row2.SRLastName and (row1.SRLastName not in overlapset))):
        return True
    return False

compare_functions = {'most_conservative': most_conservative_comparative_function,
                     'medium': medium_comparative_function}

def find_unique_number_duplicates(dfgroupbadindex, repeat_caller_conservativeness='use_all_calls', overlapsetlocal=None):
    """
    Finds the unique number duplicates in the dataframe per incident ID, i.e., removing repeat callers
    """
    global overlapset
    overlapset = overlapsetlocal

    if repeat_caller_conservativeness == 'use_all_calls':
        return dfgroupbadindex.shape[0] - 1

    if dfgroupbadindex.shape[0] == 1:
        return 0
    # for each incident, loop through the rows and compare each row to the ones before it to see if same caller
    # if same caller, don't count that row in the uniques
    dfgroup = dfgroupbadindex.reset_index()
    comp_function = compare_functions[repeat_caller_conservativeness]
    is_unique_caller_list = [True]
    for en, row1 in dfgroup.iterrows():
        if en == 0:
            continue
        isunique = True
        for en2, row2 in dfgroup.iloc[:en].iterrows():
            pairsmatch = is_unique_caller_list[en2] and comp_function(
                row1, row2)
            # above checks if row1 is same as row2, if row2 is unique (not repeat of a previous one)
            if pairsmatch:
                isunique = False
                break
        is_unique_caller_list.append(isunique)
    return sum([1 for i in is_unique_caller_list if i])