"""
General utility functions and building blocks for synthesizers
"""
import numpy as np
import pandas as pd

from diffprivlib.mechanisms import LaplaceTruncated
from thomas.core import Factor, CPT, JPT
from sys import maxsize

def dp_contingency_table(data, epsilon):
    #print("^^^^^^^^^^^^^^^^^^^^^^ inside contingency table data....",data)
    """Compute differentially private contingency table of input data"""
    #print("\n\n\n\n\n\n\n----############################################################----[DP Contingency] input data in the function\n\n\n\n\n\n\n\n")
    contingency_table_ = contingency_table(data)
    #print("temporary ^^^^^^^^^ contingenecy table values....",contingency_table_)
    # if we remove one record from X the count in one cell decreases by 1 while the rest stays the same.
    sensitivity = 1
    dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)
    #print("temporary 11111 contingenecy table values....",contingency_table_.values.flatten())
    contingency_table_values = contingency_table_.values.flatten()
    #print("temporary 22222 contingenecy table values....",contingency_table_values)
    dp_contingency_table = np.zeros_like(contingency_table_values)
    #print("temporary 33333 contingenecy table values....",dp_contingency_table)
    for i in np.arange(dp_contingency_table.shape[0]):
        # round counts upwards to preserve bins with noisy count between [0, 1]
        dp_contingency_table[i] = np.ceil(dp_mech.randomise(contingency_table_values[i]))
    #     print(f"\n^^^^^^ temporary contingenecy table values for {i} is\n",dp_contingency_table[i])
    # print("\n^^^^^^ Temp contingency table after loop:\n",dp_contingency_table)
    # print("\n^^^^^^ Temp contingency states for the table above are:\n",contingency_table_.states)
    return Factor(dp_contingency_table, states=contingency_table_.states)


# def dp_marginal_distribution(data, epsilon):
#     """Compute differentially private marginal distribution of input data"""
#     print("----START----[DP marginal] input data in the function\n")
#     marginal_ = marginal_distribution(data)
#     # print("\n#####        marginal_.values being printed in dp_marginal_distribution ....",marginal_.values) 
#     # print("\n#####        marginal_.states being printed in dp_marginal_distribution ....",marginal_.states)
#     # print("\n#####        data.shape[0 being printed in dp_marginal_distribution ....",data.shape[0])
#     # print("\n#####        data being printed in dp_marginal_distribution ....",data)
#     # print("\n#####        entire marginal_ is printed ....\n",marginal_)
    
#     # removing one record from X will decrease probability 1/n in one cell of the
#     # marginal distribution and increase the probability 1/n in the remaining cells
#     sensitivity = 2 / data.shape[0]
#     dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)
#     print("\n[DP marginal] DP mechanism used is",dp_mech)
#     marginal_values = marginal_.values.flatten()
#     print("\n[DP marginal] marginal_values are :",marginal_values)
#     dp_marginal = np.zeros_like(marginal_.values)
#     print("\n[DP marginal] new dp_marginal initialized as 0",dp_marginal,"\n")

#     for i in np.arange(dp_marginal.shape[0]):
#         # round counts upwards to preserve bins with noisy count between [0, 1]
#         dp_marginal[i] = dp_mech.randomise(marginal_.values[i])
#         print(f"[DP marginal] updated marginal values of {i} by randomizing is",dp_marginal[i],"\n")
#     dp_marginal = _normalize_distribution(dp_marginal)
#     print("\n[DP marginal] dp_marginal after being normalized in dp_marginal_distribution",dp_marginal)
#     print("\n----END----[DP marginal] returning output of this loop in dp_marginal_distribution is",Factor(dp_marginal, states=marginal_.states))
#     return Factor(dp_marginal, states=marginal_.states)


def dp_marginal_distribution(data, epsilon):
    """Compute differentially private marginal distribution of input data"""
    # print("----START----[DP marginal] input data in the function\n")
    marginal_ = marginal_distribution(data)
    # print("\n#####        marginal_.values being printed in dp_marginal_distribution ....",marginal_.values) 
    # print("\n#####        marginal_.states being printed in dp_marginal_distribution ....",marginal_.states)
    # print("\n#####        data.shape[0 being printed in dp_marginal_distribution ....",data.shape[0])
    # print("\n#####        data being printed in dp_marginal_distribution ....",data)
    # print("\n#####        entire marginal_ is printed ....\n",marginal_)
    
    # removing one record from X will decrease probability 1/n in one cell of the
    # marginal distribution and increase the probability 1/n in the remaining cells
    sensitivity = 2 / data.shape[0]
    dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)
    # print("\n[DP marginal] DP mechanism used is",dp_mech)
    marginal_values = marginal_.values.flatten()
    #print("\n[DP marginal] marginal_values are :",marginal_values)
    dp_marginal = np.zeros_like(marginal_.values)
    #print("\n[DP marginal] new dp_marginal initialized as 0", dp_marginal, "\n")
    #print("\n[DP marginal] the values being passed for the randomising....", marginal_.values)
    # Randomize entire array at once
    dp_marginal = dp_mech.randomise(marginal_.values)

    # Print updated marginal values
    #print("\n[DP marginal] the values being returned after dp_marginal randomising:", dp_marginal)

    # Normalize the distribution
    dp_marginal = _normalize_distribution(dp_marginal)

    # print("\n[DP marginal] dp_marginal after being normalized in dp_marginal_distribution",dp_marginal)
    # print("\n----END----[DP marginal] returning output of this loop in dp_marginal_distribution is",Factor(dp_marginal, states=marginal_.states))
    return Factor(dp_marginal, states=marginal_.states)






# def dp_joint_distribution(data, epsilon):
#     """Compute differentially private joint distribution of input data"""
#     print("\n----START----[DP Joint] input data in the function\n",data)
#     joint_distribution_ = joint_distribution(data)

#     # removing one record from X will decrease probability 1/n in one cell of the
#     # joint distribution and increase the probability 1/n in the remaining cells
#     sensitivity = 2 / data.shape[0]
#     dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)
#     #print("[DP Joint] joint_distribution_.values....",joint_distribution_.values)
#     joint_distribution_values = joint_distribution_.values.flatten()
#     print("\n[DP Joint] joint_distribution_values....",joint_distribution_values)
#     dp_joint_distribution_ = np.zeros_like(joint_distribution_values)
#     print("\n[DP Joint] dp_joint_distribution_ copied and initialized as 0 ....",dp_joint_distribution_,"\n")

#     for i in np.arange(dp_joint_distribution_.shape[0]):
#         dp_joint_distribution_[i] = dp_mech.randomise(joint_distribution_values[i])
#         print(f"[DP Joint] dp_joint_distribution_ at {i} is....",dp_joint_distribution_[i],"\n")
#     dp_joint_distribution_ = _normalize_distribution(dp_joint_distribution_)
#     #print("----END1----[DP Joint] dp_joint_distribution_....",dp_joint_distribution_)
#     print("----END----[DP Joint] returning the final returning value of JPT ",JPT(dp_joint_distribution_, states=joint_distribution_.states))
#     return JPT(dp_joint_distribution_, states=joint_distribution_.states)


def dp_joint_distribution(data, epsilon):
    """Compute differentially private joint distribution of input data"""
    # print("\n----START----[DP Joint] input data in the function\n",data)
    joint_distribution_ = joint_distribution(data)

    # removing one record from X will decrease probability 1/n in one cell of the
    # joint distribution and increase the probability 1/n in the remaining cells
    sensitivity = 2 / data.shape[0]
    dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)
    #print("[DP Joint] joint_distribution_.values....",joint_distribution_.values)
    joint_distribution_values = joint_distribution_.values.flatten()
    # print("\n[DP Joint] joint_distribution_values....",joint_distribution_values)
    dp_joint_distribution_ = np.zeros_like(joint_distribution_values)
    # print("\n[DP Joint] dp_joint_distribution_ copied and initialized as 0 ....",dp_joint_distribution_,"\n")
    # print("\n[DP Joint] the values being passed for the randomising....",joint_distribution_values)
    dp_joint_distribution_ = dp_mech.randomise(joint_distribution_values)
    # print(f"[DP Joint] the values being returned after the randomising........",dp_joint_distribution_,"\n")
    dp_joint_distribution_ = _normalize_distribution(dp_joint_distribution_)
    #print("----END1----[DP Joint] dp_joint_distribution_....",dp_joint_distribution_)
    # print("----END----[DP Joint] returning the final returning value of JPT ",JPT(dp_joint_distribution_, states=joint_distribution_.states))
    return JPT(dp_joint_distribution_, states=joint_distribution_.states)




# def dp_conditional_distribution(data, epsilon, conditioned=None):
#     """Compute differentially private conditional distribution of input data
#     Inferred from marginal or joint distribution"""
#     # if only one columns (series or dataframe), i.e. no conditioning columns
#     if len(data.squeeze().shape) == 1:
#         dp_distribution = dp_marginal_distribution(data, epsilon=epsilon)
#     else:
#         dp_distribution = dp_joint_distribution(data, epsilon=epsilon)
#     cpt = CPT(dp_distribution, conditioned=conditioned)

#     # normalize if cpt has conditioning columns
#     if cpt.conditioning:
#         cpt = _normalize_cpt(cpt)
#     return cpt

def dp_conditional_distribution(data, epsilon, conditioned=None):
    """Compute differentially private conditional distribution of input data
    Inferred from marginal or joint distribution"""
    # print("\n----START---- [DP Conditional] input data in the function\n",data)
    # if only one columns (series or dataframe), i.e. no conditioning columns
    #print("\n[DP Conditional] lenght of data is",len(data.squeeze().shape),"\n")
    if len(data.squeeze().shape) == 1:
        # print("\n---No conditioning columns found, means this column has no parents.!! ---> Marginal\n")
        dp_distribution = dp_marginal_distribution(data, epsilon=epsilon)
        # print("\n[DP Conditional] dp_distribution for the marginal distribution is\n",dp_distribution,"\nNext is CPT...")
    else:
        # print("\n---Conditioning columns found, means this column has parents..!! ---> Joint \n" )
        dp_distribution = dp_joint_distribution(data, epsilon=epsilon)
        # print("\n[DP Conditional] dp_distribution for the joint distribution is\n",dp_distribution,"\nNext is CPT...")
    
    cpt = CPT(dp_distribution, conditioned=conditioned)
    # print("\n[DP Conditional] output of main CPT class after normalizing the prob() \n",cpt)
    
    # normalize if cpt has conditioning columns
    if cpt.conditioning:
        # print("\n[DP Conditional] cpt has conditioning, so normalizing ..\n",cpt.conditioning)
        cpt = _normalize_cpt(cpt)
        # print("\n[DP Conditional] normalized cpt is \n",cpt)
    # print("\n----END---- [DP Conditional] final returning cpt after the checking of cpt conditioning....\n",cpt)
    return cpt


"""Non-differentially private functions below"""

def contingency_table(data):
    # print("\n----##################################----[Contingency table ] input data in the function\n",data)
    # print("\n----##################################----[Contingency table ] returns the values as below\n",Factor.from_data(data))
    return Factor.from_data(data)

def joint_distribution(data):
    """Get joint distribution by normalizing contingency table"""
    # print("\n----##############----[Joint distribution] input data in the function\n",data)
    # print("\n----##############----[Joint distribution] returns the values as below after normalizing the cpt from the contingency table\n",contingency_table(data).normalize())
    return contingency_table(data).normalize()

def marginal_distribution(data):
    assert len(data.squeeze().shape) == 1, "data can only consist of a single column"
    # converts single column dataframe to series
    data = data.squeeze()

    marginal = data.value_counts(normalize=True, dropna=False)
    #print("\n[Marginal-marg distribution] marginal being returned is",marginal)
    states = {data.name: marginal.index.tolist()}
    #print("\n[Marginal-marg distribution] states being returned is",states)
    # print("\n----END----[Marginal-marg distribution] marginal being returned is",Factor(marginal, states=states)) 
    return Factor(marginal, states=states)

def uniform_distribution(data):
    assert len(data.squeeze().shape) == 1, "data can only consist of a single column"
    # converts single column dataframe to series
    data = data.squeeze()
    n_unique = data.nunique(dropna=False)
    uniform = np.full(n_unique, 1/n_unique)
    states = {data.name: data.unique().tolist()}
    return Factor(uniform, states=states)

def compute_distribution(data):
    """"Draws a marginal or joint distribution depending on the number of input dimensions"""
    if len(data.squeeze().shape) == 1:
        return marginal_distribution(data)
    else:
        return joint_distribution(data)

"""Check and fix distributions"""


def _normalize_distribution(distribution):
    """Check whether probability distribution sums to 1"""
    # print("\n[normalize distribution] normalizing the randomized distribution, prob() sum should be 1")
    # print(" --- [normalize distribution] distribution before normalization",distribution)
    distribution = _check_all_zero(distribution)
    #print("\n Temporary,             distribution after all zero check....",distribution)
    #distribution = distribution / distribution.sum()
    #distribution = np.array(distribution) / np.sum(distribution)
    distribution_sum = sum(distribution)
    if distribution_sum != 0:
        distribution = [value / distribution_sum for value in distribution]


    
    # print(" +++ [normalize distribution] distribution after normalization",distribution)
    return distribution


def _check_all_zero(distribution):
    """In case distribution contains only zero values due to DP noise, convert to uniform"""
    if not np.any(distribution):
        #print("\n Temporary,             distribution contains only zero values....",distribution)
        distribution = np.repeat(1 / len(distribution), repeats=len(distribution))
        #print("\n Temporary,             distribution after converting to uniform....",distribution)
    #print("\n Temporary,             output returning distribution....",distribution)
    return distribution


def _normalize_cpt(cpt):
    """normalization of cpt with option to fill missing values with uniform distribution"""
    # convert to series as normalize does not work with thomas cpts
    series = cpt.as_series()
    series_norm_full = series / series.unstack().sum(axis=1)
    # fill missing combinations with uniform distribution
    uniform_prob = 1 / len(cpt.states[cpt.conditioned[-1]])
    series_norm_full = series_norm_full.fillna(uniform_prob)
    return CPT(series_norm_full, cpt.states)

def _ensure_arg_is_list(arg):
    if not arg:
        raise ValueError('Argument is empty: {}'.format(arg))

    arg = [arg] if isinstance(arg, str) else arg
    arg = list(arg) if isinstance(arg, tuple) else arg
    assert isinstance(arg, list), "input argument should be either string, tuple or list"
    return arg

def cardinality(X):
    """Compute cardinality of input data"""
    return np.prod(X.nunique(dropna=False))

def rank_columns_on_cardinality(X):
    """Rank columns based on number of unique values"""
    return X.nunique().sort_values(ascending=False)

def astype_categorical(data, include_nan=True):
    """Convert data to categorical and optionally adds nan as unique category"""
    # converts to dataframe in case of numpy input and make all columns categorical.
    data = pd.DataFrame(data).astype('category', copy=False)

    # add nan as category
    if include_nan:
        nan_columns = data.columns[data.isna().any()]
        for c in nan_columns:
            data[c] = data[c].cat.add_categories('nan').fillna('nan')
    return data











# Previous code with full debugging print statements..!!




# """
# General utility functions and building blocks for synthesizers
# """
# import numpy as np
# import pandas as pd

# from diffprivlib.mechanisms import LaplaceTruncated
# from thomas.core import Factor, CPT, JPT
# from sys import maxsize

# def dp_contingency_table(data, epsilon):
#     #print("^^^^^^^^^^^^^^^^^^^^^^ inside contingency table data....",data)
#     """Compute differentially private contingency table of input data"""
#     print("\n\n\n\n\n\n\n----############################################################----[DP Contingency] input data in the function\n\n\n\n\n\n\n\n")
#     contingency_table_ = contingency_table(data)
#     #print("temporary ^^^^^^^^^ contingenecy table values....",contingency_table_)
#     # if we remove one record from X the count in one cell decreases by 1 while the rest stays the same.
#     sensitivity = 1
#     dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)
#     #print("temporary 11111 contingenecy table values....",contingency_table_.values.flatten())
#     contingency_table_values = contingency_table_.values.flatten()
#     #print("temporary 22222 contingenecy table values....",contingency_table_values)
#     dp_contingency_table = np.zeros_like(contingency_table_values)
#     #print("temporary 33333 contingenecy table values....",dp_contingency_table)
#     for i in np.arange(dp_contingency_table.shape[0]):
#         # round counts upwards to preserve bins with noisy count between [0, 1]
#         dp_contingency_table[i] = np.ceil(dp_mech.randomise(contingency_table_values[i]))
#         print(f"\n^^^^^^ temporary contingenecy table values for {i} is\n",dp_contingency_table[i])
#     print("\n^^^^^^ Temp contingency table after loop:\n",dp_contingency_table)
#     print("\n^^^^^^ Temp contingency states for the table above are:\n",contingency_table_.states)
#     return Factor(dp_contingency_table, states=contingency_table_.states)


# # def dp_marginal_distribution(data, epsilon):
# #     """Compute differentially private marginal distribution of input data"""
# #     print("----START----[DP marginal] input data in the function\n")
# #     marginal_ = marginal_distribution(data)
# #     # print("\n#####        marginal_.values being printed in dp_marginal_distribution ....",marginal_.values) 
# #     # print("\n#####        marginal_.states being printed in dp_marginal_distribution ....",marginal_.states)
# #     # print("\n#####        data.shape[0 being printed in dp_marginal_distribution ....",data.shape[0])
# #     # print("\n#####        data being printed in dp_marginal_distribution ....",data)
# #     # print("\n#####        entire marginal_ is printed ....\n",marginal_)
    
# #     # removing one record from X will decrease probability 1/n in one cell of the
# #     # marginal distribution and increase the probability 1/n in the remaining cells
# #     sensitivity = 2 / data.shape[0]
# #     dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)
# #     print("\n[DP marginal] DP mechanism used is",dp_mech)
# #     marginal_values = marginal_.values.flatten()
# #     print("\n[DP marginal] marginal_values are :",marginal_values)
# #     dp_marginal = np.zeros_like(marginal_.values)
# #     print("\n[DP marginal] new dp_marginal initialized as 0",dp_marginal,"\n")

# #     for i in np.arange(dp_marginal.shape[0]):
# #         # round counts upwards to preserve bins with noisy count between [0, 1]
# #         dp_marginal[i] = dp_mech.randomise(marginal_.values[i])
# #         print(f"[DP marginal] updated marginal values of {i} by randomizing is",dp_marginal[i],"\n")
# #     dp_marginal = _normalize_distribution(dp_marginal)
# #     print("\n[DP marginal] dp_marginal after being normalized in dp_marginal_distribution",dp_marginal)
# #     print("\n----END----[DP marginal] returning output of this loop in dp_marginal_distribution is",Factor(dp_marginal, states=marginal_.states))
# #     return Factor(dp_marginal, states=marginal_.states)


# def dp_marginal_distribution(data, epsilon):
#     """Compute differentially private marginal distribution of input data"""
#     print("----START----[DP marginal] input data in the function\n")
#     marginal_ = marginal_distribution(data)
#     # print("\n#####        marginal_.values being printed in dp_marginal_distribution ....",marginal_.values) 
#     # print("\n#####        marginal_.states being printed in dp_marginal_distribution ....",marginal_.states)
#     # print("\n#####        data.shape[0 being printed in dp_marginal_distribution ....",data.shape[0])
#     # print("\n#####        data being printed in dp_marginal_distribution ....",data)
#     # print("\n#####        entire marginal_ is printed ....\n",marginal_)
    
#     # removing one record from X will decrease probability 1/n in one cell of the
#     # marginal distribution and increase the probability 1/n in the remaining cells
#     sensitivity = 2 / data.shape[0]
#     dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)
#     print("\n[DP marginal] DP mechanism used is",dp_mech)
#     marginal_values = marginal_.values.flatten()
#     print("\n[DP marginal] marginal_values are :",marginal_values)
#     dp_marginal = np.zeros_like(marginal_.values)
#     print("\n[DP marginal] new dp_marginal initialized as 0", dp_marginal, "\n")
#     print("\n[DP marginal] the values being passed for the randomising....", marginal_.values)
#     # Randomize entire array at once
#     dp_marginal = dp_mech.randomise(marginal_.values)

#     # Print updated marginal values
#     print("\n[DP marginal] the values being returned after dp_marginal randomising:", dp_marginal)

#     # Normalize the distribution
#     dp_marginal = _normalize_distribution(dp_marginal)

#     print("\n[DP marginal] dp_marginal after being normalized in dp_marginal_distribution",dp_marginal)
#     print("\n----END----[DP marginal] returning output of this loop in dp_marginal_distribution is",Factor(dp_marginal, states=marginal_.states))
#     return Factor(dp_marginal, states=marginal_.states)






# # def dp_joint_distribution(data, epsilon):
# #     """Compute differentially private joint distribution of input data"""
# #     print("\n----START----[DP Joint] input data in the function\n",data)
# #     joint_distribution_ = joint_distribution(data)

# #     # removing one record from X will decrease probability 1/n in one cell of the
# #     # joint distribution and increase the probability 1/n in the remaining cells
# #     sensitivity = 2 / data.shape[0]
# #     dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)
# #     #print("[DP Joint] joint_distribution_.values....",joint_distribution_.values)
# #     joint_distribution_values = joint_distribution_.values.flatten()
# #     print("\n[DP Joint] joint_distribution_values....",joint_distribution_values)
# #     dp_joint_distribution_ = np.zeros_like(joint_distribution_values)
# #     print("\n[DP Joint] dp_joint_distribution_ copied and initialized as 0 ....",dp_joint_distribution_,"\n")

# #     for i in np.arange(dp_joint_distribution_.shape[0]):
# #         dp_joint_distribution_[i] = dp_mech.randomise(joint_distribution_values[i])
# #         print(f"[DP Joint] dp_joint_distribution_ at {i} is....",dp_joint_distribution_[i],"\n")
# #     dp_joint_distribution_ = _normalize_distribution(dp_joint_distribution_)
# #     #print("----END1----[DP Joint] dp_joint_distribution_....",dp_joint_distribution_)
# #     print("----END----[DP Joint] returning the final returning value of JPT ",JPT(dp_joint_distribution_, states=joint_distribution_.states))
# #     return JPT(dp_joint_distribution_, states=joint_distribution_.states)


# def dp_joint_distribution(data, epsilon):
#     """Compute differentially private joint distribution of input data"""
#     print("\n----START----[DP Joint] input data in the function\n",data)
#     joint_distribution_ = joint_distribution(data)

#     # removing one record from X will decrease probability 1/n in one cell of the
#     # joint distribution and increase the probability 1/n in the remaining cells
#     sensitivity = 2 / data.shape[0]
#     dp_mech = LaplaceTruncated(epsilon=epsilon, lower=0, upper=maxsize, sensitivity=sensitivity)
#     #print("[DP Joint] joint_distribution_.values....",joint_distribution_.values)
#     joint_distribution_values = joint_distribution_.values.flatten()
#     print("\n[DP Joint] joint_distribution_values....",joint_distribution_values)
#     dp_joint_distribution_ = np.zeros_like(joint_distribution_values)
#     print("\n[DP Joint] dp_joint_distribution_ copied and initialized as 0 ....",dp_joint_distribution_,"\n")
#     print("\n[DP Joint] the values being passed for the randomising....",joint_distribution_values)
#     dp_joint_distribution_ = dp_mech.randomise(joint_distribution_values)
#     print(f"[DP Joint] the values being returned after the randomising........",dp_joint_distribution_,"\n")
#     dp_joint_distribution_ = _normalize_distribution(dp_joint_distribution_)
#     #print("----END1----[DP Joint] dp_joint_distribution_....",dp_joint_distribution_)
#     print("----END----[DP Joint] returning the final returning value of JPT ",JPT(dp_joint_distribution_, states=joint_distribution_.states))
#     return JPT(dp_joint_distribution_, states=joint_distribution_.states)




# # def dp_conditional_distribution(data, epsilon, conditioned=None):
# #     """Compute differentially private conditional distribution of input data
# #     Inferred from marginal or joint distribution"""
# #     # if only one columns (series or dataframe), i.e. no conditioning columns
# #     if len(data.squeeze().shape) == 1:
# #         dp_distribution = dp_marginal_distribution(data, epsilon=epsilon)
# #     else:
# #         dp_distribution = dp_joint_distribution(data, epsilon=epsilon)
# #     cpt = CPT(dp_distribution, conditioned=conditioned)

# #     # normalize if cpt has conditioning columns
# #     if cpt.conditioning:
# #         cpt = _normalize_cpt(cpt)
# #     return cpt

# def dp_conditional_distribution(data, epsilon, conditioned=None):
#     """Compute differentially private conditional distribution of input data
#     Inferred from marginal or joint distribution"""
#     print("\n----START---- [DP Conditional] input data in the function\n",data)
#     # if only one columns (series or dataframe), i.e. no conditioning columns
#     #print("\n[DP Conditional] lenght of data is",len(data.squeeze().shape),"\n")
#     if len(data.squeeze().shape) == 1:
#         print("\n---No conditioning columns found, means this column has no parents.!! ---> Marginal\n")
#         dp_distribution = dp_marginal_distribution(data, epsilon=epsilon)
#         print("\n[DP Conditional] dp_distribution for the marginal distribution is\n",dp_distribution,"\nNext is CPT...")
#     else:
#         print("\n---Conditioning columns found, means this column has parents..!! ---> Joint \n" )
#         dp_distribution = dp_joint_distribution(data, epsilon=epsilon)
#         print("\n[DP Conditional] dp_distribution for the joint distribution is\n",dp_distribution,"\nNext is CPT...")
    
#     cpt = CPT(dp_distribution, conditioned=conditioned)
#     print("\n[DP Conditional] output of main CPT class after normalizing the prob() \n",cpt)
    
#     # normalize if cpt has conditioning columns
#     if cpt.conditioning:
#         print("\n[DP Conditional] cpt has conditioning, so normalizing ..\n",cpt.conditioning)
#         cpt = _normalize_cpt(cpt)
#         print("\n[DP Conditional] normalized cpt is \n",cpt)
#     print("\n----END---- [DP Conditional] final returning cpt after the checking of cpt conditioning....\n",cpt)
#     return cpt


# """Non-differentially private functions below"""

# def contingency_table(data):
#     print("\n----##################################----[Contingency table ] input data in the function\n",data)
#     print("\n----##################################----[Contingency table ] returns the values as below\n",Factor.from_data(data))
#     return Factor.from_data(data)

# def joint_distribution(data):
#     """Get joint distribution by normalizing contingency table"""
#     print("\n----##############----[Joint distribution] input data in the function\n",data)
#     print("\n----##############----[Joint distribution] returns the values as below after normalizing the cpt from the contingency table\n",contingency_table(data).normalize())
#     return contingency_table(data).normalize()

# def marginal_distribution(data):
#     assert len(data.squeeze().shape) == 1, "data can only consist of a single column"
#     # converts single column dataframe to series
#     data = data.squeeze()

#     marginal = data.value_counts(normalize=True, dropna=False)
#     #print("\n[Marginal-marg distribution] marginal being returned is",marginal)
#     states = {data.name: marginal.index.tolist()}
#     #print("\n[Marginal-marg distribution] states being returned is",states)
#     print("\n----END----[Marginal-marg distribution] marginal being returned is",Factor(marginal, states=states)) 
#     return Factor(marginal, states=states)

# def uniform_distribution(data):
#     assert len(data.squeeze().shape) == 1, "data can only consist of a single column"
#     # converts single column dataframe to series
#     data = data.squeeze()
#     n_unique = data.nunique(dropna=False)
#     uniform = np.full(n_unique, 1/n_unique)
#     states = {data.name: data.unique().tolist()}
#     return Factor(uniform, states=states)

# def compute_distribution(data):
#     """"Draws a marginal or joint distribution depending on the number of input dimensions"""
#     if len(data.squeeze().shape) == 1:
#         return marginal_distribution(data)
#     else:
#         return joint_distribution(data)

# """Check and fix distributions"""


# def _normalize_distribution(distribution):
#     """Check whether probability distribution sums to 1"""
#     print("\n[normalize distribution] normalizing the randomized distribution, prob() sum should be 1")
#     print(" --- [normalize distribution] distribution before normalization",distribution)
#     distribution = _check_all_zero(distribution)
#     #print("\n Temporary,             distribution after all zero check....",distribution)
#     #distribution = distribution / distribution.sum()
#     #distribution = np.array(distribution) / np.sum(distribution)
#     distribution_sum = sum(distribution)
#     if distribution_sum != 0:
#         distribution = [value / distribution_sum for value in distribution]


    
#     print(" +++ [normalize distribution] distribution after normalization",distribution)
#     return distribution


# def _check_all_zero(distribution):
#     """In case distribution contains only zero values due to DP noise, convert to uniform"""
#     if not np.any(distribution):
#         #print("\n Temporary,             distribution contains only zero values....",distribution)
#         distribution = np.repeat(1 / len(distribution), repeats=len(distribution))
#         #print("\n Temporary,             distribution after converting to uniform....",distribution)
#     #print("\n Temporary,             output returning distribution....",distribution)
#     return distribution


# def _normalize_cpt(cpt):
#     """normalization of cpt with option to fill missing values with uniform distribution"""
#     # convert to series as normalize does not work with thomas cpts
#     series = cpt.as_series()
#     series_norm_full = series / series.unstack().sum(axis=1)
#     # fill missing combinations with uniform distribution
#     uniform_prob = 1 / len(cpt.states[cpt.conditioned[-1]])
#     series_norm_full = series_norm_full.fillna(uniform_prob)
#     return CPT(series_norm_full, cpt.states)

# def _ensure_arg_is_list(arg):
#     if not arg:
#         raise ValueError('Argument is empty: {}'.format(arg))

#     arg = [arg] if isinstance(arg, str) else arg
#     arg = list(arg) if isinstance(arg, tuple) else arg
#     assert isinstance(arg, list), "input argument should be either string, tuple or list"
#     return arg

# def cardinality(X):
#     """Compute cardinality of input data"""
#     return np.prod(X.nunique(dropna=False))

# def rank_columns_on_cardinality(X):
#     """Rank columns based on number of unique values"""
#     return X.nunique().sort_values(ascending=False)

# def astype_categorical(data, include_nan=True):
#     """Convert data to categorical and optionally adds nan as unique category"""
#     # converts to dataframe in case of numpy input and make all columns categorical.
#     data = pd.DataFrame(data).astype('category', copy=False)

#     # add nan as category
#     if include_nan:
#         nan_columns = data.columns[data.isna().any()]
#         for c in nan_columns:
#             data[c] = data[c].cat.add_categories('nan').fillna('nan')
#     return data