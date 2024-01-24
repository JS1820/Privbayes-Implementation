import argparse
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
from pathlib import Path
import os
import time

def main():
    parser = argparse.ArgumentParser(description='Generate and compare synthetic data using PrivBayes.')
    parser.add_argument('--dataset', type=str, help='Path to the input dataset in CSV format')
    parser.add_argument('--epsilon', type=float, default=1.5, help='Epsilon value for differential privacy')
    parser.add_argument('--rows', type=int, default=5000, help='Number of synthetic rows to generate')
    parser.add_argument('--bayesian', type=int, default=3, help='Degree of Bayesian network')
    parser.add_argument('--categorical',type=str, help='Give the names of attributes that are categorical in nature, Use Correct attribute names')
    parser.add_argument('--compare', type=str, help='Specify which comparison functions to invoke (comma-separated)')
    args = parser.parse_args()
    print("\n============================== Start ==============================\n")
    
    output_folder = f'/Privbayes-Implementation/privbayes-datasynthesizer/Output/correlated_attribute_mode/'
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    print(f"Initial contents of the output folder : {output_folder}\n")
    print("\n".join(os.listdir(output_folder)))  # Use os.listdir() instead of listdir()

    input_data = args.dataset
    epsilon = args.epsilon
    num_tuples_to_generate = args.rows
    degree_of_bayesian_network = args.bayesian

    df = pd.read_csv(input_data)
    print("\nPrinting the head of the input_df\n",df.head())

    mode = 'correlated_attribute_mode'
    description_file = f'/Privbayes-Implementation/privbayes-datasynthesizer/Output/correlated_attribute_mode/original_dataset_description.json'
    synthetic_data = f'/Privbayes-Implementation/privbayes-datasynthesizer/Output/correlated_attribute_mode/sythetic_dataset.csv'

    threshold_value = 15
    if args.categorical:
        categorical_attributes = {attr: True for attr in args.categorical.split(',')}
        print("\nCategorical attributes are :", categorical_attributes,"\n")
    else:
        categorical_attributes = {col: True for col in df.columns}

    candidate_keys = {}
    #print("\n======== Data being Described for Bayesian Network ===========\n")

    describer = DataDescriber(category_threshold=threshold_value)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,
                                                            epsilon=epsilon,
                                                            k=degree_of_bayesian_network,
                                                            attribute_to_is_categorical=categorical_attributes,
                                                            attribute_to_is_candidate_key=candidate_keys)
    describer.save_dataset_description_to_file(description_file)
    display_bayesian_network(describer.bayesian_network)
    print("\nData description file saved to : /Privbayes-implementation/privbayes-datasynthesizer/Output/corelated_attribute_mode/\n")

    print("\n================== Synthetic Data being Generated ==================\n")
    start_time = time.time()
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
    generator.save_synthetic_data(synthetic_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\nSynthetic data generation process has been completed.\n")
    print(f"\nTime taken to complete generating {args.rows} rows: {elapsed_time:.2f} seconds\n")

    
    input_df = pd.read_csv(input_data, skipinitialspace=True)
    synthetic_df = pd.read_csv(synthetic_data)
    attribute_description = read_json_file(description_file)['attribute_description']

    if args.compare and '1' in args.compare:
        comparedatasets1way(input_df,synthetic_df)
        print("into 1")
    if args.compare and '2' in args.compare:
        comparedatasets2way(input_df, synthetic_df)
        print("into 2")

    if args.compare and '3' in args.compare:
        comparedatasets3way(input_df, synthetic_df)
    print("\n============================= Completed ===============================\n")
    print(f"\nFinal contents of the output folder {output_folder}:\n")
    print("\n".join(os.listdir(output_folder)))  # Use os.listdir() instead of listdir()
    print("\n")
    print("\nPrinting the head of the Synthetic dataset generated :\n",synthetic_df.head())

def comparedatasets1way(input_df, synthetic_df):
    print(f"\n\n========== Comparing the datasets using 1 way occurrences ==========\n\n")

    # Get columns (attributes) from both original and synthetic datasets
    original_columns = input_df.columns.tolist()
    synthetic_columns = synthetic_df.columns.tolist()

    # Find common attributes between original and synthetic datasets
    common_attributes = set(original_columns) & set(synthetic_columns)

    # Create subplots for attribute comparison
    fig, axs = plt.subplots(len(common_attributes), 1, figsize=(10, 5 * len(common_attributes)))

    for i, attribute in enumerate(common_attributes):
        # Calculate the proportions of occurrences for each unique value in input_df
        input_attribute_proportions = input_df[attribute].value_counts(normalize=True).reset_index(name='Proportion')

        # Calculate the proportions of occurrences for each unique value in synthetic_df
        synthetic_attribute_proportions = synthetic_df[attribute].value_counts(normalize=True).reset_index(name='Proportion')

        # Combine unique values from both datasets
        unique_values = set(input_attribute_proportions['index']).union(set(synthetic_attribute_proportions['index']))

        # Merge the proportions for each unique value
        merged_proportions = pd.merge(input_attribute_proportions, synthetic_attribute_proportions,
                                      how='outer', on='index', suffixes=('_InputDF', '_SyntheticDF')).fillna(0)

        # Plotting the proportions of occurrences as bar plots for input_df and synthetic_df
        axs[i].bar(merged_proportions['index'], merged_proportions['Proportion_InputDF'], color='blue', label='Input DF')
        axs[i].bar(merged_proportions['index'], merged_proportions['Proportion_SyntheticDF'], color='red', alpha=0.5, label='Synthetic DF')
        axs[i].set_xlabel(attribute)
        axs[i].set_ylabel('Proportion of Occurrences')
        axs[i].set_title(f'Comparison of {attribute} between Input DF and Synthetic DF')
        axs[i].tick_params(axis='x', rotation=90)
        axs[i].legend()

    plt.tight_layout()
    plt.show()
    print("\n\n\n\n")


def comparedatasets2way(input_df, synthetic_df):
    print(f"\n\n========== Comparing the datasets using 2-way occurrences ==========\n\n")

    # Get columns (attributes) from both original and synthetic datasets
    original_columns = input_df.columns.tolist()
    synthetic_columns = synthetic_df.columns.tolist()

    # Find common attributes between original and synthetic datasets
    common_attributes = set(original_columns) & set(synthetic_columns)

    # Generate attribute pairs for the common attributes
    attribute_pairs = list(itertools.combinations(common_attributes, 2))

    # Create subplots for co-occurrence graphs for each attribute pair
    fig, axs = plt.subplots(min(len(attribute_pairs), 25), 1, figsize=(10, 8 * min(len(attribute_pairs), 25)))

    filtered_attribute_pairs = []

    for i, attribute_pair in enumerate(attribute_pairs):
        # Calculate the relative occurrences for each unique pair of values in input_df
        input_attribute_counts = input_df.groupby(list(attribute_pair)).size() / len(input_df)

        # Calculate the relative occurrences for each unique pair of values in synthetic_df
        synthetic_attribute_counts = synthetic_df.groupby(list(attribute_pair)).size() / len(synthetic_df)

        # Combine and get the highest and lowest values
        combined_values = pd.concat([input_attribute_counts, synthetic_attribute_counts], axis=0)
        highest_value = combined_values.max()
        lowest_value = combined_values.min()

        # Calculate the median
        median_value = combined_values.median()

        # Check if the highest value is greater than the median
        if highest_value > median_value:
            # Check the difference between input_df and synthetic_df
            value_difference = abs(input_attribute_counts - synthetic_attribute_counts).max()

            # Store the value difference for sorting
            filtered_attribute_pairs.append((attribute_pair, value_difference))

    # Sort attribute pairs based on value difference
    filtered_attribute_pairs.sort(key=lambda x: x[1])

    # Select the first 25 attribute pairs or all if less than 25
    selected_attribute_pairs = filtered_attribute_pairs[:min(25, len(filtered_attribute_pairs))]

    for i, (attribute_pair, _) in enumerate(selected_attribute_pairs):
        input_attribute_counts = input_df.groupby(list(attribute_pair)).size() / len(input_df)
        synthetic_attribute_counts = synthetic_df.groupby(list(attribute_pair)).size() / len(synthetic_df)

        input_attribute_counts.plot(kind='bar', color='blue', alpha=0.7, label='Input DF', ax=axs[i])
        synthetic_attribute_counts.plot(kind='bar', color='red', alpha=0.5, label='Synthetic DF', ax=axs[i])

        axs[i].set_xlabel(f"{attribute_pair[0]} - {attribute_pair[1]}")
        axs[i].set_ylabel('Relative Occurrences')
        axs[i].set_title(f"Relative Co-occurrence of {attribute_pair[0]} and {attribute_pair[1]}")
        axs[i].tick_params(axis='x', rotation=90)
        axs[i].legend()

    plt.tight_layout()
    plt.show()
    print("\n\n\n\n")

def comparedatasets3way(input_df, synthetic_df):
    print(f"\n\n========== Comparing the datasets using 3-way occurrences ==========\n\n")

    # Get columns (attributes) from both original and synthetic datasets
    original_columns = input_df.columns.tolist()
    synthetic_columns = synthetic_df.columns.tolist()

    # Find common attributes between original and synthetic datasets
    common_attributes = set(original_columns) & set(synthetic_columns)

    # Generate attribute trios for the common attributes
    attribute_trios = list(itertools.combinations(common_attributes, 3))

    # Create subplots for co-occurrence graphs for each attribute trio
    fig, axs = plt.subplots(min(len(attribute_trios), 25), 1, figsize=(10, 8 * min(len(attribute_trios), 25)))

    filtered_attribute_trios = []

    for i, attribute_trio in enumerate(attribute_trios):
        # Calculate the relative occurrences for each unique trio of values in input_df
        input_attribute_counts = input_df.groupby(list(attribute_trio)).size() / len(input_df)

        # Calculate the relative occurrences for each unique trio of values in synthetic_df
        synthetic_attribute_counts = synthetic_df.groupby(list(attribute_trio)).size() / len(synthetic_df)

        # Combine and get the highest and lowest values
        combined_values = pd.concat([input_attribute_counts, synthetic_attribute_counts], axis=0)
        highest_value = combined_values.max()
        lowest_value = combined_values.min()

        # Calculate the median
        median_value = combined_values.median()

        # Check if the highest value is greater than the median
        if highest_value > median_value:
            # Check the difference between input_df and synthetic_df
            value_difference = abs(input_attribute_counts - synthetic_attribute_counts).max()

            # Store the value difference for sorting
            filtered_attribute_trios.append((attribute_trio, value_difference))

    # Sort attribute trios based on value difference
    filtered_attribute_trios.sort(key=lambda x: x[1])

    # Select the first 25 attribute trios or all if less than 25
    selected_attribute_trios = filtered_attribute_trios[:min(25, len(filtered_attribute_trios))]

    for i, (attribute_trio, _) in enumerate(selected_attribute_trios):
        input_attribute_counts = input_df.groupby(list(attribute_trio)).size() / len(input_df)
        synthetic_attribute_counts = synthetic_df.groupby(list(attribute_trio)).size() / len(synthetic_df)

        input_attribute_counts.plot(kind='bar', color='blue', alpha=0.7, label='Input DF', ax=axs[i])
        synthetic_attribute_counts.plot(kind='bar', color='red', alpha=0.5, label='Synthetic DF', ax=axs[i])

        axs[i].set_xlabel(f"{attribute_trio[0]} - {attribute_trio[1]} - {attribute_trio[2]}")
        axs[i].set_ylabel('Relative Occurrences')
        axs[i].set_title(f"Relative Co-occurrence of {attribute_trio[0]}, {attribute_trio[1]}, and {attribute_trio[2]}")
        axs[i].tick_params(axis='x', rotation=90)
        axs[i].legend()

    plt.tight_layout()
    plt.show()
    print("\n\n\n\n")


if __name__ == "__main__":
    main()


# import argparse
# import itertools
# import matplotlib.pyplot as plt
# import pandas as pd
# from DataSynthesizer.DataDescriber import DataDescriber
# from DataSynthesizer.DataGenerator import DataGenerator
# from DataSynthesizer.ModelInspector import ModelInspector
# from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
# from pathlib import Path
# import os

# def main():
#     parser = argparse.ArgumentParser(description='Generate and compare synthetic data using PrivBayes.')
#     parser.add_argument('--dataset', type=str, help='Path to the input dataset in CSV format')
#     parser.add_argument('--epsilon', type=float, default=0, help='Epsilon value for differential privacy')
#     parser.add_argument('--syntheticrows', type=int, default=5000, help='Number of synthetic rows to generate')
#     parser.add_argument('--bayesian', type=int, default=3, help='Degree of Bayesian network')
#     parser.add_argument('--compare', nargs='+', type=int, default=[1], help='Comparison type(s)')
#     args = parser.parse_args()
    
#     output_folder = f'/Privbayes-Implementation/privbayes-datasynthesizer/Output/correlated_attribute_mode/'
#     Path(output_folder).mkdir(parents=True, exist_ok=True)
#     print(f"Initial contents of the output folder {output_folder}:\n")
#     print("\n".join(os.listdir(output_folder)))  # Use os.listdir() instead of listdir()

#     input_data = args.dataset
#     epsilon = args.epsilon
#     num_tuples_to_generate = args.syntheticrows
#     degree_of_bayesian_network = args.bayesian

#     df = pd.read_csv(input_data)
#     print(df.head())

#     mode = 'correlated_attribute_mode'
#     description_file = f'/Privbayes-Implementation/privbayes-datasynthesizer/Output/correlated_attribute_mode/original_dataset_description.json'
#     synthetic_data = f'/Privbayes-Implementation/privbayes-datasynthesizer/Output/correlated_attribute_mode/sythetic_dataset.csv'

#     threshold_value = 15
#     categorical_attributes = {'education': True, 'marital-status': True}
#     candidate_keys = {}

#     describer = DataDescriber(category_threshold=threshold_value)
#     describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,
#                                                             epsilon=epsilon,
#                                                             k=degree_of_bayesian_network,
#                                                             attribute_to_is_categorical=categorical_attributes,
#                                                             attribute_to_is_candidate_key=candidate_keys)
#     describer.save_dataset_description_to_file(description_file)
#     display_bayesian_network(describer.bayesian_network)

#     generator = DataGenerator()
#     generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
#     generator.save_synthetic_data(synthetic_data)

#     input_df = pd.read_csv(input_data, skipinitialspace=True)
#     synthetic_df = pd.read_csv(synthetic_data)
#     attribute_description = read_json_file(description_file)['attribute_description']

#     comparison_types = args.compare
#     for compare_type in comparison_types:
#         if compare_type == 1:
#             comparedatasets1way(input_df,synthetic_df,attribute_description)
#         elif compare_type == 2:
#             comparedatasets2way(input_df, synthetic_df)
#         elif compare_type == 3:
#             comparedatasets3way(input_df, synthetic_df)

#     print(f"\nFinal contents of the output folder {output_folder}:\n")
#     print("\n".join(os.listdir(output_folder)))  # Use os.listdir() instead of listdir()

# def comparedatasets1way(input_df,synthetic_df,attribute_description):
#     from DataSynthesizer.ModelInspector import ModelInspector
#     inspector = ModelInspector(input_df, synthetic_df, attribute_description)
#     print(f"\n\n=========================COMPARING THE DATASETS USING 1 WAY OCCURRENCES==============================\n\n")

#     for attribute in synthetic_df.columns:
#         inspector.compare_histograms(attribute)
#     print("\n\n\n\n")


# def comparedatasets2way(input_df, synthetic_df):
#     # Assuming you have your input_df and synthetic_df already loaded
    
#     # Get columns (attributes) from both original and synthetic datasets
#     original_columns = input_df.columns.tolist()
#     synthetic_columns = synthetic_df.columns.tolist()
    
#     # Find common attributes between original and synthetic datasets
#     common_attributes = set(original_columns) & set(synthetic_columns)
    
#     # Generate attribute pairs for the common attributes
#     attribute_pairs = list(itertools.combinations(common_attributes, 2))
    
#     # Define a minimum threshold for co-occurrence counts
#     min_threshold = 20  # Adjust this threshold as needed
    
#     # Initialize variables to keep track of maximum counts in both dataframes
#     max_input_count = 0
#     max_synthetic_count = 0
    
#     # Create subplots for co-occurrence graphs for each attribute pair
#     fig, axs = plt.subplots(len(attribute_pairs), 1, figsize=(10, 5 * len(attribute_pairs)), sharex=True, gridspec_kw={'hspace': 0.5})
    
#     for i, attribute_pair in enumerate(attribute_pairs):
#         # Calculate the counts of occurrences for each unique pair of values in input_df
#         input_attribute_counts = input_df.groupby(list(attribute_pair)).size().reset_index(name='Count')
    
#         # Calculate the counts of occurrences for each unique pair of values in synthetic_df
#         synthetic_attribute_counts = synthetic_df.groupby(list(attribute_pair)).size().reset_index(name='Count')
    
#         # Filter insignificant co-occurrences based on the threshold
#         input_attribute_counts = input_attribute_counts[input_attribute_counts['Count'] >= min_threshold]
#         synthetic_attribute_counts = synthetic_attribute_counts[synthetic_attribute_counts['Count'] >= min_threshold]
    
#         # Update the maximum counts
#         max_input_count = max(max_input_count, input_attribute_counts['Count'].max())
#         max_synthetic_count = max(max_synthetic_count, synthetic_attribute_counts['Count'].max())
    
#         # Plotting the counts of co-occurrences as bar plots for input_df and synthetic_df
#         axs[i].bar(input_attribute_counts.apply(lambda x: f"{x[attribute_pair[0]]} - {x[attribute_pair[1]]}", axis=1),
#                    input_attribute_counts['Count'], color='blue', label='Input DF')
#         axs[i].bar(synthetic_attribute_counts.apply(lambda x: f"{x[attribute_pair[0]]} - {x[attribute_pair[1]]}", axis=1),
#                    synthetic_attribute_counts['Count'], color='red', alpha=0.5, label='Synthetic DF')
#         axs[i].set_xlabel(f"{attribute_pair[0]} - {attribute_pair[1]}")
#         axs[i].set_ylabel('Count')
#         axs[i].set_title(f"Co-occurrence of {attribute_pair[0]} and {attribute_pair[1]}")
#         axs[i].tick_params(axis='x', rotation=90)
#         axs[i].set_ylim(0, max(max_input_count, max_synthetic_count))
#         axs[i].legend()
    
#     plt.tight_layout()
#     plt.show()


# def comparedatasets3way(input_df, synthetic_df):
#     # Assuming you have your input_df and synthetic_df already loaded
    
#     # Get columns (attributes) from both original and synthetic datasets
#     original_columns = input_df.columns.tolist()
#     synthetic_columns = synthetic_df.columns.tolist()
    
#     # Find common attributes between original and synthetic datasets
#     common_attributes = set(original_columns) & set(synthetic_columns)
    
#     # Generate attribute trios for the common attributes
#     attribute_trios = list(itertools.combinations(common_attributes, 3))
    
#     # Define a minimum threshold for co-occurrence counts
#     min_threshold = 20  # Adjust this threshold as needed
    
#     # Initialize variables to keep track of maximum counts in both dataframes
#     max_input_count = 0
#     max_synthetic_count = 0
    
#     # Create subplots for co-occurrence graphs for each attribute trio
#     fig, axs = plt.subplots(len(attribute_trios), 1, figsize=(10, 5 * len(attribute_trios)), sharex=True, gridspec_kw={'hspace': 0.5})
    
#     for i, attribute_trio in enumerate(attribute_trios):
#         # Calculate the counts of occurrences for each unique trio of values in input_df
#         input_attribute_counts = input_df.groupby(list(attribute_trio)).size().reset_index(name='Count')
    
#         # Calculate the counts of occurrences for each unique trio of values in synthetic_df
#         synthetic_attribute_counts = synthetic_df.groupby(list(attribute_trio)).size().reset_index(name='Count')
    
#         # Filter insignificant co-occurrences based on the threshold
#         input_attribute_counts = input_attribute_counts[input_attribute_counts['Count'] >= min_threshold]
#         synthetic_attribute_counts = synthetic_attribute_counts[synthetic_attribute_counts['Count'] >= min_threshold]
    
#         # Update the maximum counts
#         max_input_count = max(max_input_count, input_attribute_counts['Count'].max())
#         max_synthetic_count = max(max_synthetic_count, synthetic_attribute_counts['Count'].max())
    
#         # Plotting the counts of co-occurrences as bar plots for input_df and synthetic_df
#         axs[i].bar(input_attribute_counts.apply(lambda x: f"{x[attribute_trio[0]]} - {x[attribute_trio[1]]} - {x[attribute_trio[2]]}", axis=1),
#                    input_attribute_counts['Count'], color='blue', label='Input DF')
#         axs[i].bar(synthetic_attribute_counts.apply(lambda x: f"{x[attribute_trio[0]]} - {x[attribute_trio[1]]} - {x[attribute_trio[2]]}", axis=1),
#                    synthetic_attribute_counts['Count'], color='red', alpha=0.5, label='Synthetic DF')
#         axs[i].set_xlabel(f"{attribute_trio[0]} - {attribute_trio[1]} - {attribute_trio[2]}")
#         axs[i].set_ylabel('Count')
#         axs[i].set_title(f"Co-occurrence of {attribute_trio[0]}, {attribute_trio[1]}, and {attribute_trio[2]}")
#         axs[i].tick_params(axis='x', rotation=90)
#         axs[i].set_ylim(0, max(max_input_count, max_synthetic_count))
#         axs[i].legend()
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()
