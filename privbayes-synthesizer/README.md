# PrivBayes via Synthesizer

This repository contains the implementation of PrivBayes via synthesizer, a differentially private synthetic data generation algorithm. PrivBayes via synthesizer generates synthetic data that preserves the statistical properties of the original dataset while providing privacy guarantees through differential privacy.

## Getting Started

### Prerequisites

- Python 3.x
- pandas
- numpy
- argparse

### Installation

1. Clone the repository:

    ```shell
    git clone https://github.com/0hex7/privbayes-implementation.git
    ```

2. Install the required dependencies:

    ```shell
    pip install pandas numpy argparse
    ```

## Usage

1. Prepare your input dataset in CSV format and place it in the `input-data` directory.

2. Run the `privbayes.py` script with the desired command-line arguments:

    ```shell
    python privbayes.py --dataset <path-to-dataset> --bucket <bucket-size> --epsilon <privacy-epsilon>
    ```

    - `--dataset`: Path to the input dataset file (default: `../input-data/adult.csv`).
    - `--bucket`: Size of the buckets for numerical values (default: `10`).
    - `--epsilon`: Epsilon value for differential privacy (default: `1.0`).

3. The script will preprocess the dataset, generate a synthetic dataset using PrivBayes via synthesizer, and perform various evaluations on the synthetic dataset.

## Optimization

The code has been optimized to ensure efficient execution. By implementing various techniques, we have achieved the ability to generate over 40,000 records within 7 seconds. This optimization allows for faster data generation and analysis, making PrivBayes via synthesizer a powerful tool for privacy-preserving data synthesis.

## Bayesian Network

In addition to generating records, PrivBayes via synthesizer also constructs a Bayesian network. The Bayesian network captures the probabilistic relationships between different attributes in the generated data. This network can be used for various purposes, such as data analysis, inference, and decision-making.

## Results

The script will output the following results:

- The head of the preprocessed original dataset.
- The number of rows in the dataset.
- Attribute Parent Pairs development progress.
- Time taken to generate the synthetic dataset.
- The head of the synthetic dataset.
- Marginal comparison score and graph.
- Association comparison score and graph.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request. We value the input of the community and strive to make PrivBayes via synthesizer better with each contribution.

## Conclusion

In conclusion, PrivBayes via synthesizer provides a powerful and efficient solution for privacy-preserving data synthesis. With its optimization techniques, it can generate over 40,000 records within 7 seconds, making it suitable for large-scale data generation tasks. The accompanying Bayesian network further enhances the utility of the generated data. We encourage you to explore PrivBayes via synthesizer and contribute to its development.
