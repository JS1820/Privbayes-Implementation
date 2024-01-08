# Privbayes-Implementation: Privacy-Preserving Data Release Using Bayesian Networks

## Overview

PrivBayes-Implementation is a comprehensive repository that extensively covers the implementation of PrivBayes, a cutting-edge methodology designed to facilitate the release of private datasets while ensuring individual data privacy. The core concept involves leveraging Bayesian Networks to generate synthetic data that mimics the statistical properties of the original dataset while protecting individual privacy through local differentiation techniques.

## Source Paper: PrivBayes - A Privacy-Preserving Data Release Framework

The main reference paper detailing the PrivBayes algorithm is titled "PrivBayes: Private Data Release via Bayesian Networks", authored by Jun Zhang, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, and Xiaokui Xiao. Published in the ACM Transactions on Database Systems (TODS) in 2017 (Volume 42, Issue 4, Article 25, 41 pages), the paper introduces the PrivBayes algorithm as a robust framework for releasing private data while preserving the confidentiality of individuals' sensitive information. The DOI for the paper is [10.1145/3134428](https://doi.org/10.1145/3134428).

## Introduction: PrivBayes - A Revolutionary Approach to Data Privacy

PrivBayes represents a groundbreaking solution aimed at securely releasing private datasets. The algorithm operates by generating synthetic data using Bayesian Networks. This synthesized data retains similar statistical properties to the original dataset while ensuring individual privacy through various privacy-preserving mechanisms. The implementation focuses on locally differentiating privacy levels across records while maintaining the utility of the released dataset.

## File Structure: Organized Repository Architecture

The repository's file structure is meticulously organized:

- **code**: Contains the central implementation of the PrivBayes algorithm.
  - **util**: Houses essential utility functions utilized within the primary functions.

- **data**: Manages dataset and output file storage.
  - **preprocessed-outputs**: Stores preprocessed data in categorical format.
  - **synthetic-outputs**: Contains categorized synthetic datasets pre-postprocessing.
  - **postprocessed-outputs**: Holds data post-de-categorization, restoring synthetic datasets to their original format.
  - **graphs**: Stores graphical representations of datasets (original and synthetic), illustrating pairwise attribute occurrences.
  - **outputs**: Contains output images from various test runs, providing an overview of expected results.
  - **research-paper**: Houses the referenced research paper.

## Datasets: Exploring Available Data for Experimentation

The repository offers a variety of datasets. Notably, experiments primarily focus on 'adult.csv' and 'adult_tiny.csv'. Users are encouraged to explore and test other datasets available in the /data/ folder to diversify experimental scenarios.

## Requirements: Essential Dependencies for Execution

All necessary requirements are listed in the 'requirements.txt' file. These requirements are automatically installed during the execution of 'setup.py'.

## Pre-execution Checklist: Ensuring Prerequisites for Running the Functions

Before executing the main functions, it's crucial to ensure the installation of [ektelo](https://github.com/ektelo/ektelo) and [autograd](https://github.com/HIPS/autograd).

## Execution: Running PrivBayes for Privacy-Preserving Data Release

Execute 'privbayes.py' to initiate PrivBayes. Input the desired dataset name for processing (from /data/ folder). Upon completion, a graphical representation comparing the two-way occurrences of attributes in the original and synthetic datasets will be stored in /data/graphs/.

## Acknowledgments: Recognizing the Contributions and Support

The original rights to the PrivBayes paper belong to its authors. The code in this repository references various sources and undergoes numerous tweaks and reconfigurations.

## Contributors: Collaborative Efforts Behind This Implementation

- [Arun Ashok Badri](https://github.com/0hex7/) : iamarunbadri@gmail.com
- [Sandhya V](https://github.com/sandxxax/) : vsandhya2912@gmail.com
- [Pavan Kumar J](https://github.com/Lonelypheonix/) : pavankumarj.cy@gmail.com

## References: Citing the Source Paper

The referenced paper for PrivBayes:
Zhang, J., Cormode, G., Procopiuc, C. M., Srivastava, D., & Xiao, X. (2017). PrivBayes: Private Data Release via Bayesian Networks. ACM Transactions on Database Systems (TODS), 42(4), Article 25, 41 pages. https://doi.org/10.1145/3134428

Feel free to expand or modify any section further to suit your requirements.
