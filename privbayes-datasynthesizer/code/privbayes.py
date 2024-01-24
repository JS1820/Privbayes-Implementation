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

def main():
    parser = argparse.ArgumentParser(description='Generate and compare synthetic data using PrivBayes.')
    parser.add_argument('--dataset', type=str, help='Path to the input dataset in CSV format')
    parser.add_argument('--epsilon', type=float, default=0, help='Epsilon value for differential privacy')
    parser.add_argument('--syntheticrows', type=int, default=5000, help='Number of synthetic rows to generate')
    parser.add_argument('--bayesian', type=int, default=3, help='Degree of Bayesian network')
    args = parser.parse_args()
    
    output_folder = f'/Privbayes-Implementation/privbayes-datasynthesizer/Output/correlated_attribute_mode/'
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    print(f"Initial contents of the output folder {output_folder}:\n")
    print("\n".join(os.listdir(output_folder)))  # Use os.listdir() instead of listdir()

    
    input_data = args.dataset
    epsilon = args.epsilon
    num_tuples_to_generate = args.syntheticrows
    degree_of_bayesian_network = args.bayesian

    df = pd.read_csv(input_data)
    print(df.head())

    mode = 'correlated_attribute_mode'
    description_file = f'/Privbayes-Implementation/privbayes-datasynthesizer/Output/correlated_attribute_mode/original_dataset_description.json'
    synthetic_data = f'/Privbayes-Implementation/privbayes-datasynthesizer/Output/correlated_attribute_mode/sythetic_dataset.csv'

    threshold_value = 15
    categorical_attributes = {'education': True, 'marital-status': True}
    candidate_keys = {}

    describer = DataDescriber(category_threshold=threshold_value)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,
                                                            epsilon=epsilon,
                                                            k=degree_of_bayesian_network,
                                                            attribute_to_is_categorical=categorical_attributes,
                                                            attribute_to_is_candidate_key=candidate_keys)
    describer.save_dataset_description_to_file(description_file)
    display_bayesian_network(describer.bayesian_network)

    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
    generator.save_synthetic_data(synthetic_data)

    input_df = pd.read_csv(input_data, skipinitialspace=True)
    synthetic_df = pd.read_csv(synthetic_data)
    attribute_description = read_json_file(description_file)['attribute_description']

    inspector = ModelInspector(input_df, synthetic_df, attribute_description)
    comparedatasets(input_df, synthetic_df)

    
    print(f"\nFinal contents of the output folder {output_folder}:\n")
    print("\n".join(listdir(output_folder)))

def comparedatasets(input_df, synthetic_df):
  original_columns = input_df.columns.tolist()
  synthetic_columns = synthetic_df.columns.tolist()

  # Find common attributes between original and synthetic datasets
  common_attributes = set(original_columns) & set(synthetic_columns)

  # Generate attribute pairs for the common attributes
  attribute_pairs = list(itertools.combinations(common_attributes, 2))

  # Define a minimum threshold for co-occurrence counts
  min_threshold = 30  # Adjust this threshold as needed

  # Initialize variables to keep track of maximum counts in both dataframes
  max_input_count = 0
  max_synthetic_count = 0

  # Create subplots for co-occurrence graphs for each attribute pair
  fig, axs = plt.subplots(len(attribute_pairs), 2, figsize=(15, 5 * len(attribute_pairs)))

  for i, attribute_pair in enumerate(attribute_pairs):
      # Calculate the counts of occurrences for each unique pair of values in input_df
      input_attribute_counts = input_df.groupby(list(attribute_pair)).size().reset_index(name='Count')

      # Calculate the counts of occurrences for each unique pair of values in synthetic_df
      synthetic_attribute_counts = synthetic_df.groupby(list(attribute_pair)).size().reset_index(name='Count')

      # Filter insignificant co-occurrences based on the threshold
      input_attribute_counts = input_attribute_counts[input_attribute_counts['Count'] >= min_threshold]
      synthetic_attribute_counts = synthetic_attribute_counts[synthetic_attribute_counts['Count'] >= min_threshold]

      # Update the maximum counts
      max_input_count = max(max_input_count, input_attribute_counts['Count'].max())
      max_synthetic_count = max(max_synthetic_count, synthetic_attribute_counts['Count'].max())

      # Plotting the counts of co-occurrences as bar plots for input_df and synthetic_df
      axs[i, 0].bar(input_attribute_counts.apply(lambda x: f"{x[attribute_pair[0]]} - {x[attribute_pair[1]]}", axis=1), input_attribute_counts['Count'])
      axs[i, 0].set_xlabel(f"{attribute_pair[0]} - {attribute_pair[1]}")
      axs[i, 0].set_ylabel('Count')
      axs[i, 0].set_title(f"Co-occurrence of {attribute_pair[0]} and {attribute_pair[1]} in input_df")
      axs[i, 0].tick_params(axis='x', rotation=90)
      axs[i, 0].set_ylim(0, max(max_input_count, max_synthetic_count))
      axs[i, 0].set_xlim(-1, len(input_attribute_counts))

      axs[i, 1].bar(synthetic_attribute_counts.apply(lambda x: f"{x[attribute_pair[0]]} - {x[attribute_pair[1]]}", axis=1), synthetic_attribute_counts['Count'])
      axs[i, 1].set_xlabel(f"{attribute_pair[0]} - {attribute_pair[1]}")
      axs[i, 1].set_ylabel('Count')
      axs[i, 1].set_title(f"Co-occurrence of {attribute_pair[0]} and {attribute_pair[1]} in synthetic_df")
      axs[i, 1].tick_params(axis='x', rotation=90)
      axs[i, 1].set_ylim(0, max(max_input_count, max_synthetic_count))
      axs[i, 1].set_xlim(-1, len(synthetic_attribute_counts))

  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
    main()
