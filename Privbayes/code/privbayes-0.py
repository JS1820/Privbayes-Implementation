from ektelo.algorithm.privBayes import privBayesSelect
import numpy as np
from util import Dataset, Factor, FactoredInference, mechanism
from ektelo.matrix import Identity
import pandas as pd
import itertools
import argparse
import benchmarks
import os
import json
import sys


"""
This file implements PrivBayes.

Zhang, Jun, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, and Xiaokui Xiao. "Privbayes: Private data release via bayesian networks." ACM Transactions on Database Systems (TODS) 42, no. 4 (2017): 25.
"""

def get_dataset_file(dataset_name):
    dataset_folder = '/privbayes-implementation/Privbayes/data/'
    csv_file = f'{dataset_folder}{dataset_name}.csv'
    
    # Check if the CSV file exists
    if os.path.isfile(csv_file):
        return csv_file
    else:
        print(f"{dataset_name}.csv not found.")
        sys.exit(1)



def preprocess(original_dataset):
    # Load your dataset
    data = pd.read_csv(original_dataset)

    # Extract file name from the path (without extension)
    file_name = os.path.splitext(os.path.basename(original_dataset))[0]

    # Output directory path
    output_directory = '/privbayes-implementation/Privbayes/data/processed-output/'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Output file names
    processed_output_file = output_directory + f'processed_{file_name}.csv'
    domain_output_file = output_directory + f'domain_{file_name}.json'

    # Check if processed output file exists, if yes, remove it
    if os.path.isfile(processed_output_file):
        os.remove(processed_output_file)

    # Check if domain output file exists, if yes, remove it
    if os.path.isfile(domain_output_file):
        os.remove(domain_output_file)

    # Categorize columns and generate domain values
    processed_data, domain_values = categorize_columns(data)

    print("[+] Data Categorization is completed")

    # Save the processed dataset with categorization to the output directory
    processed_data.to_csv(processed_output_file, index=False)

    # Save domain values as a JSON file in the output directory
    with open(domain_output_file, 'w') as json_file:
        json.dump(domain_values, json_file)

    # Return the processed dataset file path and domain file path
    return processed_output_file, domain_output_file, file_name


def categorize_columns(data):
    # Initialize an empty dictionary to store domain values for each column
    domain_values = {}

    # Loop through each column in the dataset
    for column in data.columns:
        # Create a mapping of unique categorical values to numeric labels
        unique_values = data[column].unique()
        value_mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}

        # Map categorical values to numeric labels in the dataset
        data[column] = data[column].map(value_mapping)

        # Store the number of unique values for the domain of this column
        domain_values[column] = len(value_mapping)

    return data, domain_values




def privbayes_measurements(data, eps=1.0, seed=0):
    
    domain = data.domain
    config = ''
    for a in domain:
        values = [str(i) for i in range(domain[a])]
        config += 'D ' + ' '.join(values) + ' \n'
    config = config.encode('utf-8')
    
    values = np.ascontiguousarray(data.df.values.astype(np.int32))
    print(values)
    print("12. back to main func loaded successfully")
   
    ans = privBayesSelect.py_get_model(values, config, eps/2, 1.0, seed)
    print("13. back to main func loaded successfully")
    
    ans = ans.decode('utf-8')[:-1]
    print("14. back to main func loaded successfully")

    projections = []
    for m in ans.split('\n'):
        p = [domain.attrs[int(a)] for a in m.split(',')[::2]]
        projections.append(tuple(p))
    print("15. back to main func loaded successfully")
  
    prng = np.random.RandomState(seed) 
    measurements = []
    delta = len(projections)
    for proj in projections:
        x = data.project(proj).datavector()
        I = Identity(x.size)
        y = I.dot(x) + prng.laplace(loc=0, scale=4*delta/eps, size=x.size)
        measurements.append( (I, y, 1.0, proj) )
    print("16. back to main func loaded successfully")
     
    return measurements

def privbayes_inference(domain, measurements, total, file_name):
    file_name = file_name
    synthetic = pd.DataFrame()
    print(measurements[0])
    _, y, _, proj = measurements[0]
    y = np.maximum(y, 0)
    y /= y.sum()
    col = proj[0]
    synthetic[col] = np.random.choice(domain[col], total, True, y)
        
    for _, y, _, proj in measurements[1:]:
        # find the CPT
        col, dep = proj[0], proj[1:]
        print(col)
        y = np.maximum(y, 0)
        dom = domain.project(proj)
        cpt = Factor(dom, y.reshape(dom.shape))
        marg = cpt.project(dep)
        cpt /= marg
        print(cpt)
        cpt2 = np.moveaxis(cpt.project(proj).values, 0, -1)
        print(cpt2)
        
        # sample current column
        synthetic[col] = 0
        rng = itertools.product(*[range(domain[a]) for a in dep])
        for v in rng:
            idx = (synthetic.loc[:,dep].values == np.array(v)).all(axis=1)
            p = cpt2[v].flatten()
            if p.sum() == 0:
                p = np.ones(p.size) / p.size
            n = domain[col]
            N = idx.sum()
            if N > 0:
                synthetic.loc[idx,col] = np.random.choice(n, N, True, p)
    
    
    output_folder = '/privbayes-implementation/Privbayes/data/synthetic-output/'  # Define the output folder path
    output_filename = f"synthetic_{file_name}.csv"  # Define the output filename

    # Combine output folder path and output filename
    output_path = os.path.join(output_folder, output_filename)

    # Check if the output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if the file already exists, remove it
    if os.path.exists(output_path):
        os.remove(output_path)

    # Save synthetic dataset as CSV file in the specified output path
    synthetic.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset: {output_path}")
    return Dataset(synthetic, domain)

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = 'adult'
    params['iters'] = 10000
    params['epsilon'] = 1.0
    params['seed'] = 0

    return params

if __name__ == '__main__':
    dataset_name = input("Enter the dataset name: ").strip()
    
    if not dataset_name:
        print("No dataset name entered. Exiting.")
        sys.exit(1)
    
    original_dataset = get_dataset_file(dataset_name)
    print("=============DATA PRE-PROCESSING=============")
    processed_data, data_domain, file_name = preprocess(original_dataset)
    print("=============DATA PRE-PROCESSING COMPLETED=============")
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', choices=['adult'], help='dataset to use')
    parser.add_argument('--iters', type=int, help='number of optimization iterations')
    parser.add_argument('--epsilon', type=float, help='privacy  parameter')
    parser.add_argument('--seed', type=int, help='random seed')
    print("next step is done")

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    print("data set is being loaded into the benchmarks")

    data, workload = benchmarks.adult_benchmark(processed_data, data_domain)
    print("6. back to main func loaded successfully")
    
    total = data.df.shape[0]
    print("7. back to main func loaded successfully")

    measurements = privbayes_measurements(data, 1.0, args.seed) 
    print("8. back to main func loaded successfully")

    est = privbayes_inference(data.domain, measurements, total=total, file_name=file_name)

    elim_order = [m[3][0] for m in measurements][::-1]

    projections = [m[3] for m in measurements]
    est2, _, _ = mechanism.run(data, projections, eps=args.epsilon, frequency=50, seed=args.seed, iters=args.iters)

    def err(true, est):

        return np.sum(np.abs(true - est)) / true.sum()

    err_pb = []
    err_pgm = []
    for p, W in workload:
        true = W.dot(data.project(p).datavector())
        pb = W.dot(est.project(p).datavector())
        pgm = W.dot(est2.project(p).datavector())
        err_pb.append(err(true, pb))
        err_pgm.append(err(true, pgm))

    print('Error of PrivBayes    : %.3f' % np.mean(err_pb))
    print('Error of PrivBayes+PGM: %.3f' % np.mean(err_pgm))
