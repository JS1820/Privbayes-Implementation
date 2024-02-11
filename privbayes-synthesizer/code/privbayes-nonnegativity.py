import numpy as np
import pandas as pd
import random
import dill
from sklearn.metrics import mutual_info_score
from collections import namedtuple
from joblib import Parallel, delayed
import cProfile
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

import synthesis.synthesizers.utils as utils
from synthesis.synthesizers._base import BaseDPSynthesizer
from thomas.core import BayesianNetwork
from diffprivlib.mechanisms import Exponential
from synthesis.evaluation.evaluator import SyntheticDataEvaluator
from synthesis.evaluation.metrics import MarginalComparison, AssociationsComparison
from synthesis.evaluation.efficacy import ClassifierComparison
from synthesis.evaluation._base import BaseMetric
from synthesis.evaluation.evaluator import DEFAULT_METRICS, SyntheticDataEvaluator

APPair = namedtuple('APPair', ['attribute', 'parents'])




class PrivBayes(BaseDPSynthesizer):
    """
    Implementation of the PrivBayes algorithm for differentially private data synthesis.
    Inherits from the BaseDPSynthesizer class.
    """

    def __init__(self, epsilon=1.0, theta_usefulness=4, epsilon_split=0.3,
                 score_function='R', network_init=None, n_cpus=None, verbose=True):
        """
        Initializes the PrivBayes synthesizer with the specified hyperparameters.
        """
        super().__init__(epsilon=epsilon, verbose=verbose)
        self.theta_usefulness = theta_usefulness
        self.epsilon_split = epsilon_split  # also called Beta in paper
        self.score_function = score_function  # choose between 'R' and 'MI'
        self.network_init = network_init
        self.n_cpus = n_cpus

    def fit(self, data):
        """
        Fits the PrivBayes synthesizer to the input data.
        """
        data = self._check_input_data(data)
        self._check_init_args()

        self._greedy_bayes(data)
        self._compute_conditional_distributions(data)
        self.model_ = BayesianNetwork.from_CPTs('PrivBayes', self.cpt_.values())
        return self

    def _check_init_args(self):
        """
        Checks the validity of the initialization arguments.
        """
        super()._check_init_args()
        self._check_score_function()

    def _check_score_function(self):
        """
        Checks the input score function and sets the sensitivity.
        """
        if self.score_function.upper() not in ['R', 'MI']:
            raise ValueError("Score function must be 'R' or 'MI'")

        if self.score_function.upper() == 'R':
            self._score_sensitivity = (3 / self.n_records_fit_) + (2 / self.n_records_fit_**2)
        elif self.score_function.upper() == 'MI':
            self._score_sensitivity = (2 / self.n_records_fit_) * np.log((self.n_records_fit_ + 1) / 2) + \
                              (((self.n_records_fit_ - 1) / self.n_records_fit_) *
                               np.log((self.n_records_fit_ + 1) / (self.n_records_fit_ - 1)))

    def sample(self, n_records=None):
        """
        Generates synthetic data using the PrivBayes synthesizer.
        """
        self._check_is_fitted()
        n_records = n_records or self.n_records_fit_

        data_synth = self._generate_data(n_records)
        data_synth = self._check_output_data(data_synth)

        if self.verbose:
            print("\n[+] Synthetic Dataset is Generated\n")
        return data_synth

    def _greedy_bayes(self, data):
        """
        Performs the greedy Bayesian network construction algorithm.
        """
        nodes, nodes_selected = self._init_network(data)

        # normally len(nodes) - 1, unless user initialized part of the network
        self._n_nodes_dp_computed = len(nodes) - len(nodes_selected)

        for i in range(len(nodes_selected), len(nodes)):
            if self.verbose:
                print("[*] {}/{} - Evaluating next attribute to add to network".format(i + 1, len(self.columns_)))

            nodes_remaining = nodes - nodes_selected

            # select ap_pair candidates
            ap_pairs = []
            for node in nodes_remaining:
                max_domain_size = self._max_domain_size(data, node)
                max_parent_sets = self._max_parent_sets(data, nodes_selected, max_domain_size)

                # empty set - domain size of node violates theta_usefulness
                if len(max_parent_sets) == 0:
                    ap_pairs.append(APPair(node, parents=None))
                # [empty set] - no parents found that meets domain size restrictions
                elif len(max_parent_sets) == 1 and len(max_parent_sets[0]) == 0:
                    ap_pairs.append(APPair(node, parents=None))
                else:
                    ap_pairs.extend([
                        APPair(node, parents=tuple(p)) for p in max_parent_sets
                    ])
            if self.verbose:
                print("-- Number of AttributeParentPair candidates: {}".format(len(ap_pairs)))

            scores = self._compute_scores(data, ap_pairs)
            sampled_pair = self._exponential_mechanism(ap_pairs, scores)

            if self.verbose:
                print("-- Selected attribute: '{}' - with parents: {}\n".format(sampled_pair.attribute, sampled_pair.parents))
            nodes_selected.add(sampled_pair.attribute)
            self.network_.append(sampled_pair)
        if self.verbose:
            print("\n[+] Learned Network Structure is being developed...")
        return self

    def _max_domain_size(self, data, node):
        """Computes the maximum domain size a node can have to satisfy theta-usefulness"""
        node_cardinality = utils.cardinality(data[node])
        max_domain_size = (self.n_records_fit_ * (1 - self.epsilon_split) * self.epsilon) / \
                          (2 * len(self.columns_) * self.theta_usefulness * node_cardinality)
        return max_domain_size

    def _max_parent_sets(self, data, v, max_domain_size):
        """Refer to algorithm 5 in paper - max parent set is 1) theta-useful and 2) maximal."""
        if max_domain_size < 1:
            return set()
        if len(v) == 0:
            return [set()]

        x = np.random.choice(tuple(v))
        x_domain_size = utils.cardinality(data[x])
        x = {x}

        v_without_x = v - x

        parent_sets1 = self._max_parent_sets(data, v_without_x, max_domain_size)
        parent_sets2 = self._max_parent_sets(data, v_without_x, max_domain_size / x_domain_size)

        for z in parent_sets2:
            if z in parent_sets1:
                parent_sets1.remove(z)
            parent_sets1.append(z.union(x))
        return parent_sets1

    def _init_network(self, X):
        self._binary_columns = [c for c in X.columns if X[c].unique().size <= 2]
        nodes = set(X.columns)

        if self.network_init is not None:
            nodes_selected = set(n.attribute for n in self.network_init)
            print("Pre-defined network init: {}".format(self.network_))
            for i, pair in enumerate(self.network_init):
                if self.verbose:
                    print("{}/{} - init node {} - with parents: {}".format(i + 1, len(nodes),
                                                                           pair.attribute, pair.parents))
            self.network_ = self.network_init.copy()
            return nodes, nodes_selected

        # if set_network is not called we start with a random first node
        self.network_ = []
        nodes_selected = set()

        root = np.random.choice(tuple(nodes))
        self.network_.append(APPair(attribute=root, parents=None))
        nodes_selected.add(root)
        if self.verbose:
            print("[*] 1/{} - Root of network: {}\n".format(X.shape[1], root))
        return nodes, nodes_selected

    def set_network(self, network):
        assert [isinstance(n, APPair) for n in network], "input network does not consists of APPairs"
        self.network_init = network
        return self

    def _compute_scores(self, data, ap_pairs):
        """Compute score for all ap_pairs"""
        if self.n_cpus:
            scores = Parallel(n_jobs=self.n_cpus)(delayed(self.r_score)(data, pair.attribute, pair.parents) for pair in ap_pairs)
        else:
            scores = [self.r_score(data, pair.attribute, pair.parents) for pair in ap_pairs]
        return scores

    def _exponential_mechanism(self, ap_pairs, scores):
        """select APPair with exponential mechanism"""
        local_epsilon = self.epsilon * self.epsilon_split / self._n_nodes_dp_computed
        #print("@@@@@@@@local_epsilon: ",local_epsilon)
        dp_mech = Exponential(epsilon=local_epsilon, sensitivity=self._score_sensitivity,
                              utility=list(scores), candidates=ap_pairs)
        #print("@@@@@@@@dp_mech: ",dp_mech)
        sampled_pair = dp_mech.randomise()
        #print("@@@@@@@@sampled_pair: ",sampled_pair)
        return sampled_pair
    
    
    
#####    #! debug original _compute_conditional_distributions function
    
    def _compute_conditional_distributions(self, data):
            """
            Compute the conditional distributions for each attribute in the network.
            Uses differential privacy to ensure privacy guarantees.
            
            Args:
                data: The input data used to compute the conditional distributions.
            
            Returns:
                self: The updated object with computed conditional distributions.
            """
            self.cpt_ = dict()

            local_epsilon = self.epsilon * (1 - self.epsilon_split) / len(self.columns_)
            # print("\n----START----[main code] The local_epsilon is: ",local_epsilon)
            #print("\n[+]The Attribute-Parent Pairs developed are :\n",self.network_,"\n")
            for idx, pair in enumerate(self.network_):
                if pair.parents is None:
                    attributes = [pair.attribute]
                else:
                    attributes = [*pair.parents, pair.attribute]

                cpt_size = utils.cardinality(data[attributes])
                if self.verbose:
                    print('-- Learning conditional probabilities: {} - with parents {} '
                        '~ estimated size: {}'.format(pair.attribute, pair.parents, cpt_size))

                dp_cpt = utils.dp_conditional_distribution(data[attributes], epsilon=local_epsilon)
                # print("\n[main code] The dp_cpt is: ",dp_cpt)
                self.cpt_[pair.attribute] = dp_cpt
            # print("\n[main code]The self.cpt_ is: \n",self.cpt_)
            # print("\n[main code]The entire self value is: ",self)
            # print("\n\n\n\n----END----[main code]\n\nPrinting the entire dictionory of the self\n\n",self.__dict__,"\n\n\n\n")

            return self



    # def _compute_conditional_distributions(self, data):

    #     self.cpt_ = dict()

    #     local_epsilon = self.epsilon * (1 - self.epsilon_split) / len(self.columns_)

    #     for idx, pair in enumerate(self.network_):
    #         if pair.parents is None:
    #             attributes = [pair.attribute]
    #         else:
    #             attributes = [*pair.parents, pair.attribute]

    #         # Discretize numerical attributes into buckets of size 10
    #         for attr in attributes:
    #             if pd.api.types.is_numeric_dtype(data[attr]):  # Check if the attribute is numerical
    #                 data[attr] = pd.cut(data[attr], bins=10, labels=False, precision=1)

    #                 print("\n[+] The numerical attribute is: ",attr)
    #                 print("\n[+] The discretized attribute is: ",data[attr])
    #                 print("\n[+] The discretized attribute type is: ",data[attr].dtype)
    #                 print("\n[+] The discretized attribute unique values are: ",data[attr].unique())
    #                 print("\n[+] The discretized attribute unique values count is: ",data[attr].unique().size)
    #         cpt_size = utils.cardinality(data[attributes])
    #         if self.verbose:
    #             print('Learning conditional probabilities: {} - with parents {} '
    #                 '~ estimated size: {}'.format(pair.attribute, pair.parents, cpt_size))

    #         dp_cpt = utils.dp_conditional_distribution(data[attributes], epsilon=local_epsilon)
    #         self.cpt_[pair.attribute] = dp_cpt

    #     return self


    def _generate_data(self, n_records = 30000):
        """
        Generate synthetic data based on the learned network and conditional distributions.
        
        Args:
            n_records: The number of records to generate (default: 30000).
        
        Returns:
            data_synth: The generated synthetic data.
        """
        data_synth = np.empty([n_records, len(self.columns_)], dtype=object)

        for i in range(n_records):
            # Generate a record by sampling from each attribute's conditional distribution
            for j, pair in enumerate(self.network_):
                if pair.parents is None:
                    attributes = [pair.attribute]
                else:
                    attributes = [*pair.parents, pair.attribute]
                
                # Sample from the conditional distribution
                sample = utils.sample_from_distribution(self.cpt_[pair.attribute], data_synth[i, attributes])
                
                # Assign the sampled value to the corresponding attribute in the generated record
                data_synth[i, pair.attribute] = sample
        
        return data_synth



#### #! debug the below is the original code

    def _generate_data(self, n_records = 30000):
        data_synth = np.empty([n_records, len(self.columns_)], dtype=object)

        for i in range(n_records):
            if self.verbose:
                print("\r", end='')
                print('Number of records generated: {} / {}'.format(i + 1, n_records), end='', flush=True)
            record = self._sample_record()
            data_synth[i] = list(record.values())
            # numpy.array to pandas.DataFrame with original column ordering
        data_synth = pd.DataFrame(data_synth, columns=[c.attribute for c in self.network_])
        return data_synth
###### till here is the original code


    


##### #! debug till here is the updated code..!


    # def _sample_record(self):
    #     """samples a value column for column by conditioning for parents"""
    #     record = {}
    #     for col_idx, pair in enumerate(self.network_):
    #         node = self.model_[pair.attribute]
    #         node_cpt = node.cpt
    #         node_states = node.states

    #         if node.conditioning:
    #             parent_values = [record[p] for p in node.conditioning]
    #             node_probs = node_cpt[tuple(parent_values)]

    #         else:
    #             node_probs = node_cpt.values
    #         # use random.choices over np.random.choice as np coerces, e.g. sample['nan', 1, 3.0] -> '1' (int to string)
    #         sampled_node_value = random.choices(node_states, weights=node_probs, k=1)[0] # returns list

    #         record[node.name] = sampled_node_value
    #     return record
    
    
    """Below is the main optimized version which shortens timespan by almost 15 secs for 30000 records"""


    import numpy as np

    def _sample_record(self):
        record = {}
        for col_idx, pair in enumerate(self.network_):
            node = self.model_[pair.attribute]
            node_cpt = node.cpt
            node_states = node.states

            if node.conditioning:
                parent_values = [record[p] for p in node.conditioning]
                node_probs = node_cpt[tuple(parent_values)]
            else:
                node_probs = node_cpt.values

            # Use numpy.random.choice for faster random sampling
            sampled_node_value = np.random.choice(node_states, p=node_probs)
            record[node.name] = sampled_node_value

        return record
        
        
#### 10 sec optimization 

    """def _sample_record(self):
        record = {}
        for col_idx, pair in enumerate(self.network_):
            node = self.model_[pair.attribute]
            node_cpt = node.cpt
            node_states = node.states

            if node.conditioning:
                parent_values = [record[p] for p in node.conditioning]
                node_probs = node_cpt[tuple(parent_values)]

            else:
                node_probs = node_cpt.values

            # Use NumPy for weighted random sampling
            sampled_node_value = np.random.choice(node_states, p=node_probs)

            record[node.name] = sampled_node_value
        #print(record)
        return record"""


#### 10 sec optimization
    """def _sample_record(self):

        record = {}
        for col_idx, pair in enumerate(self.network_):
            node = self.model_[pair.attribute]
            node_cpt = node.cpt
            node_states = node.states

            # Pre-calculate parent values for conditional nodes
            if node.conditioning:
                parent_values = tuple([record[p] for p in node.conditioning])
            else:
                parent_values = None

            # Access conditional probabilities efficiently
            node_probs = node_cpt[parent_values] if parent_values is not None else node_cpt.values

            # Leverage NumPy for optimized sampling
            sampled_node_value = np.random.choice(node_states, p=node_probs)
            record[node.name] = sampled_node_value

        return record"""
        
        
#### The code below uses caches mechanism, it reduces time to less than 10 seconds, but the error is too much, and the synthetic data records are not as expected.       
    """import numpy as np
    from functools import lru_cache  # For caching

    @lru_cache(maxsize=None)  # Cache function results for unchanged inputs
    def _sample_record(self):

        record = {}
        for col_idx, pair in enumerate(self.network_):
            node = self.model_[pair.attribute]
            node_cpt = node.cpt
            node_states = node.states

            # Pre-calculate parent values (potentially vectorized)
            if node.conditioning:
                parent_values = tuple([record[p] for p in node.conditioning])
                # Explore vectorizing parent_value calculations if applicable
            else:
                parent_values = None

            # Access conditional probabilities efficiently
            node_probs = node_cpt[parent_values] if parent_values is not None else node_cpt.values

            # Leverage NumPy for optimized sampling
            sampled_node_value = np.random.choice(node_states, p=node_probs)
            record[node.name] = sampled_node_value

        return record"""
        

    


    @staticmethod
    def mi_score(data, columns_a, columns_b):
        columns_a = utils._ensure_arg_is_list(columns_a)
        columns_b = utils._ensure_arg_is_list(columns_b)

        data_a = data[columns_a].squeeze()
        if len(columns_b) == 1:
            data_b = data[columns_b].squeeze()
        else:
            data_b = data.loc[:, columns_b].apply(lambda x: ' '.join(x.values), axis=1).squeeze()
        return mutual_info_score(data_a, data_b)

    @staticmethod
    def mi_score_thomas(data, columns_a, columns_b):
        columns_a = utils._ensure_arg_is_list(columns_a)
        columns_b = utils._ensure_arg_is_list(columns_b)

        prob_a = utils.compute_distribution(data[columns_a])
        prob_b = utils.compute_distribution(data[columns_b])
        prob_joint = utils.compute_distribution(data[columns_a + columns_b])

        # todo: pull-request thomas to add option to normalize to remove 0's
        # align
        prob_div = prob_joint / (prob_b * prob_a)
        prob_joint, prob_div = prob_joint.extend_and_reorder(prob_joint, prob_div)

        # remove zeros as this will result in issues with log
        prob_joint = prob_joint.values[prob_joint.values != 0]
        prob_div = prob_div.values[prob_div.values != 0]
        mi = np.sum(prob_joint * np.log(prob_div))
        # mi = np.sum(p_nodeparents.values * np.log(p_nodeparents / (p_parents * p_node)))
        return mi


    @staticmethod
    def r_score(data, columns_a, columns_b):
        """An alternative score function to mutual information with lower sensitivity - can be used on non-binary domains.
        Relies on the L1 distance from a joint distribution to a joint distributions that minimizes mutual information.
        Refer to Lemma 5.2
        """
        if columns_b is None:
            return 0
        columns_a = utils._ensure_arg_is_list(columns_a)
        columns_b = utils._ensure_arg_is_list(columns_b)

        # compute distribution that minimizes mutual information
        prob_a = utils.compute_distribution(data[columns_a])
        prob_b = utils.compute_distribution(data[columns_b])
        prob_independent = prob_b * prob_a

        # compute joint distribution
        prob_joint = utils.joint_distribution(data[columns_a + columns_b])

        # substract not part of thomas - need to ensure alignment
        # todo: should be part of thomas - submit pull-request to thomas
        prob_joint, prob_independent = prob_joint.extend_and_reorder(prob_joint, prob_independent)
        l1_distance = 0.5 * np.sum(np.abs(prob_joint.values - prob_independent.values))
        return l1_distance

    def save(self, path):
        """
        Save this synthesizer instance to the given path using pickle.

        Parameters
        ----------
        path: str
            Path where the synthesizer instance is saved.
        """
        # todo issue can't save if model is fitted - likely error within thomas
        if hasattr(self, 'model_'):
            pb = self.copy()
            del pb.model_

        with open(path, 'wb') as output:
            dill.dump(pb, output)

    @classmethod
    def load(cls, path):
        """Load a synthesizer instance from specified path.
        Parameters
        ----------
        path: str
            Path where the synthesizer instance is saved.

        Returns
        -------
        synthesizer : class
            Returns synthesizer instance.
        """
        with open(path, 'rb') as f:
            pb = dill.load(f)

        # recreate model_ attribute based on fitted network and cpt's
        if hasattr(pb, 'cpt_'):
            pb.model_ = BayesianNetwork.from_CPTs('PrivBayes', pb.cpt_.values())
        return pb



class PrivBayesFix(PrivBayes):
    """Extension to PrivBayes class to allow user to fix pre-sampled columns.
    Can be used to generate additional items for an already released synthetic dataset.
    """

    def __init__(self, epsilon=1.0, theta_usefulness=4, epsilon_split=0.3,
                 score_function='R', network_init=None, n_cpus=None, verbose=True):
        super().__init__(epsilon=epsilon, theta_usefulness=theta_usefulness, epsilon_split=epsilon_split,
                         score_function=score_function, network_init=network_init, n_cpus=n_cpus, verbose=verbose)

        if network_init is not None:
            self.network_init = network_init
            self._init_network()

    # def _init_network(self, X):
    #     nodes = set(X.columns)

    #     if self.network_init is not None:
    #         nodes_selected = set(n.attribute for n in self.network_init)
    #         print("Pre-defined network init: {}".format(self.network_))
    #         for i, pair in enumerate(self.network_init):
    #             if self.verbose:
    #                 print("{}/{} - init node {} - with parents: {}".format(i + 1, len(nodes),
    #                                                                     pair.attribute, pair.parents))
    #         self.network_ = self.network_init.copy()
    #         return nodes, nodes_selected

    #     # if set_network is not called, we start with a random first node
    #     self.network_ = []
    #     nodes_selected = set()

    #     root = np.random.choice(tuple(nodes))
    #     self.network_.append(APPair(attribute=root, parents=None))
    #     nodes_selected.add(root)
    #     if self.verbose:
    #         print("1/{} - Root of network: {}\n".format(X.shape[1], root))
    #     return nodes, nodes_selected
    
    
    def _init_network(self, X):
        nodes = set(X.columns)

        if self.network_init is not None:
            nodes_selected = set(n.attribute for n in self.network_init)
            if self.verbose:
                print("Pre-defined network init: {}".format(self.network_init))
            self.network_ = self.network_init.copy()
            return nodes, nodes_selected

        # If set_network is not called, we start with a random first node
        self.network_ = []
        nodes_selected = set()

        root = np.random.choice(tuple(nodes))
        self.network_.append(APPair(attribute=root, parents=None))
        nodes_selected.add(root)
        if self.verbose:
            print("1/{} - Root of network: {}\n".format(X.shape[1], root))
        return nodes, nodes_selected


    def fit(self, data):
        super().fit(data)
        return self

    def sample_remaining_columns(self, fixed_data):
        self._check_is_fitted()
        fixed_data = self._check_fixed_data(fixed_data)

        data_synth = self._generate_data(fixed_data)
        data_synth = self._check_output_data(data_synth)

        if self.verbose:
            print("\nRemaining synthetic columns generated")
        return data_synth

    def _generate_data(self, fixed_data):
        n_records = fixed_data.shape[0]
        data_synth = np.empty([n_records, len(self.columns_)], dtype=object)

        for i in range(n_records):
            if self.verbose:
                print("\r", end='')
                print('Number of records generated: {} / {}'.format(i + 1, n_records), end='', flush=True)
            print("\n")

            record_init = fixed_data.loc[i].to_dict()
            record = self._sample_record(record_init)
            data_synth[i] = list(record.values())

        # numpy.array to pandas.DataFrame with original column ordering
        data_synth = pd.DataFrame(data_synth, columns=[c.attribute for c in self.network_])
        return data_synth

    # def _sample_record(self, record_init):
    #     # assume X has columns with values that correspond to the first nodes in the network
    #     # that we would like to fix and condition for.
    #     record = record_init

    #     # sample remaining nodes after fixing for input X
    #     for col_idx, pair in enumerate(self.network_[len(record_init):]):
    #         node = self.model_[pair.attribute]

    #         # specify pre-sampled conditioning values
    #         node_cpt = node.cpt
    #         node_states = node.states

    #         if node.conditioning:
    #             parent_values = [record[p] for p in node.conditioning]
    #             node_probs = node_cpt[tuple(parent_values)]
    #         else:
    #             node_probs = node_cpt.values
    #         # sampled_node_value = np.random.choice(node_states, p=node_probs)
    #         sampled_node_value = random.choices(node_states, weights=node_probs)
    #         record[node.name] = sampled_node_value
    #     return record
    
    
    
    def _sample_record(self, record_init):
        record = record_init.copy()

        # sample remaining nodes after fixing for input X
        for col_idx, pair in enumerate(self.network_[len(record_init):]):
            node = self.model_[pair.attribute]

            # specify pre-sampled conditioning values
            node_cpt = node.cpt
            node_states = node.states

            if node.conditioning:
                parent_values = [record[p] for p in node.conditioning]
                node_probs = node_cpt[tuple(parent_values)]
            else:
                node_probs = node_cpt.values

            # Use np.random.choice to avoid ValueError when weights don't sum to 1
            sampled_node_value = np.random.choice(node_states, p=node_probs)
            record[node.name] = sampled_node_value.item()  # Convert to scalar

        return record

    def _check_fixed_data(self, data):
        """Checks whether the columns in fixed data where also seen in fit"""
        if isinstance(data, pd.Series):
            data = data.to_frame()

        if not np.all([c in self.columns_ for c in data.columns]):
            raise ValueError('Columns in fixed data not seen in fit.')
        if set(data.columns) == set(self.columns_):
            raise ValueError('Fixed data already contains all the columns that were seen in fit.')

        if self.verbose:
            sample_columns = [c for c in self.columns_ if c not in data.columns]
            print('Columns sampled and added to fixed data: {}'.format(sample_columns))
        # prevent integer column indexing issues
        data.columns = data.columns.astype(str)
        # make all columns categorical.
        data = data.astype('category')
        return data

class PrivBayesNP(PrivBayes):
    """Privbayes class with infinite-differential privacy, while still using epsilon value to limit the size of the network
    """

    def __init__(self, epsilon=1, theta_usefulness=4, epsilon_split=0.5,
                 score_function='R', network_init=None, n_cpus=None, verbose=True):
        self.epsilon1 = float(np.inf)
        self.epsilon2 = epsilon
        super().__init__(epsilon=self.epsilon1, theta_usefulness=theta_usefulness, epsilon_split=epsilon_split,
                         score_function=score_function, network_init=network_init, n_cpus=n_cpus, verbose=verbose)


    def _max_domain_size(self, data, node):
        """Computes the maximum domain size a node can have to satisfy theta-usefulness"""
        node_cardinality = utils.cardinality(data[node])
        max_domain_size = self.n_records_fit_ * self.epsilon2 / \
                          (2 * len(self.columns_) * self.theta_usefulness * node_cardinality)
        return max_domain_size

import time

# def preprocess_dataset(input_data):
#     print("\n[+] Preprocessing the dataset to convert numerical values into buckets of size 10 :")
#     # Remove rows with missing values
#     input_data = input_data.dropna()

#     processed_data = input_data.copy()
#     bucketized_columns = []

#     for column in input_data.columns:
#         if pd.api.types.is_numeric_dtype(input_data[column]):
#             # Process numerical columns into buckets of size 10
#             bucketized_values = pd.cut(input_data[column], bins=10, labels=False, precision=1)
#             # Replace original numerical values with bucketized values in the same column
#             processed_data[column] = bucketized_values
#             # Print information about bucketized column
#             print(f"\n[+] {column} is numerical. Bucketized values:")
#             #print(bucketized_values.value_counts().sort_index())
#             print(f"-- Domain size for {column}: {len(bucketized_values.unique())}")
#             # Add the column to the list of bucketized columns
#             bucketized_columns.append(column)
        
#     return processed_data, bucketized_columns


import pandas as pd
import numpy as np

import pandas as pd

def preprocess_dataset(input_data, bins=10):
    print(f"\n[+] Preprocessing the dataset to convert numerical values into buckets of size {bins}:")
    # Remove rows with missing values
    input_data = input_data.dropna()

    bucket_mappings = {}  # Dictionary to store mapping between original values and bucketized values

    # Column names with the same value for all records and the corresponding value
    columns_to_remove = {}

    for column in input_data.columns:
        if pd.api.types.is_numeric_dtype(input_data[column]):
            # Process numerical columns into buckets of size 10
            bucketized_values, mapping = pd.cut(input_data[column], bins, labels=False, precision=0, retbins=True)
            # Replace original numerical values with bucketized values in the same column
            input_data[column] = bucketized_values.astype(int)
            bucket_mappings[column] = mapping.astype(int)
            # Print information about bucketized column
            print(f"\n[+] {column} is numerical. Bucketized values:")
            # print(bucketized_values.value_counts().sort_index())
            print(f"-- Domain size for {column}: {len(bucketized_values.unique())}")
            # print("\n")
        # Check if all values in the column are the same
        unique_values = input_data[column].unique()
        if len(unique_values) == 1:
            print(f"\n[*] Column '{column}' has the same value ({unique_values[0]}) for all records. Removing from dataset.")
            columns_to_remove[column] = unique_values[0]

    # Remove columns with the same value for all records
    input_data.drop(columns=list(columns_to_remove.keys()), inplace=True)
    
    print("-- Columns removed: ",columns_to_remove,"\n")

    return input_data, bucket_mappings, columns_to_remove



def postprocess_dataset(data, bucket_mappings, columns_removed):
    converted_data = data.copy()

    # Restore removed columns with the same value for all records
    for column, value in columns_removed.items():
        converted_data[column] = value

    for column, mapping in bucket_mappings.items():
        if column in converted_data.columns:
            # Replace bucketized values with the average of neighboring array elements
            bucketized_values = converted_data[column]
            original_values = np.zeros_like(bucketized_values)

            for i, value in enumerate(bucketized_values):
                if np.isfinite(value):
                    lower_index = int(value) - 1  # Adjust for zero-based indexing
                    upper_index = int(value)
                    average = (mapping[lower_index] + mapping[upper_index]) / 2
                    original_values[i] = int(average)  # Convert to integer directly
                else:
                    original_values[i] = -1  # Default for non-finite values

            converted_data[column] = original_values

    return converted_data



import argparse

if __name__ == '__main__':
    
    # """Main function to demonstrate the usage of PrivBayes and PrivBayesFix."""
    # data_path = '../input-data/adult.csv'
    # data_original = pd.read_csv(data_path, engine='python')
    # # columns = ['age', 'sex', 'education', 'workclass', 'income']
    # # data = data.loc[:, columns]
    

    parser = argparse.ArgumentParser(description="Optimized PrivBayes main function")
    parser.add_argument("--dataset", type=str, default = '../input-data/adult.csv' ,help="Path to the input dataset file")
    parser.add_argument("--bucket", type=int, default=10, help="Size of the buckets for numerical values")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon value for differential privacy")
    
    
    args = parser.parse_args()

    data_path = args.dataset
    bucket = args.bucket
    epsilon = args.epsilon
    
    data_original = pd.read_csv(data_path, engine='python')   
    data,bucketized_columns,removed_columns = preprocess_dataset(data_original, bucket)
    # print(bucketized_columns,"\nbucketized_columns")
    print("\n[+] Head of the 'original' preprocessed dataset is: ")
    print(data.head())
    rows = data.shape[0]
    print("\n[+] Number of rows in the dataset is: ",rows)
    
    print("\n[+] Attribute Parent Pairs (AP Pairs) are being developed ...\n")
    pb = PrivBayes(epsilon, n_cpus=6, score_function='R', verbose=True)
    pb.fit(data)

    starttime = time.time()
    # Concatenate the results after all processes are completed
    df_synth_original = pb.sample()
    endtime = time.time()
    print(f"[+] Time taken to generate {rows} records' synthetic dataset is: {endtime - starttime}\n\n")
    df_synth = postprocess_dataset(df_synth_original, bucketized_columns,removed_columns)
    
    print("\n[+] Head of the synthetic dataset is:\n")
    print(df_synth.head())
    # result4 = pb.score(data, df_synth, score_dict=True)
    # print("\n[+] Score of the synthetic dataset is: \n",result4)

    # """test scoring functions"""
    # pair = pb.network_[3]
    # result1 = pb.mi_score(data, pair.attribute, pair.parents)
    # print("\n[+] Mutual Information Score: ", result1)
    # result2 = pb.mi_score_thomas(data, pair.attribute, pair.parents)
    # print("\n[+] Mutual Information Score Thomas: ", result2)
    # result3 = pb.r_score(data, pair.attribute, pair.parents)
    # print("\n[+] R Score: ", result3)
    
    
    
    marginal_comparison= MarginalComparison().fit(data, df_synth_original)
    print("\n[+] Marginal comparision score is: ",marginal_comparison.score())
    marginal_comparison.plot(bucketized_columns)
    plt.savefig('Marginal_comparision.pdf')
    #plt.show()
    #plt.close()
    print("-- Marginal Comparison Done & Graph is saved as a pdf file")
    
    
    association_comparison= AssociationsComparison().fit(data, df_synth_original)
    print("\n[+] Association comparision score is: ",association_comparison.score())
    association_comparison.plot()
    plt.savefig('Association_comparision.pdf')
    #plt.show()
    #plt.close()  
    print("-- Association Comparison Done & Graph is saved as a pdf file")
    print("\n[+] Completed, Exiting...\n")  



# import sys
# if __name__ == '__main__':
#     # Save the original stdout
#     original_stdout = sys.stdout
#     # Open a file in write mode to save the output
#     with open('Console-output-Nonnegativity.txt', 'w') as f:
#         # Redirect stdout to the file
#         sys.stdout = f

#         parser = argparse.ArgumentParser(description="Optimized PrivBayes main function")
#         parser.add_argument("--dataset", type=str, default='../input-data/adult.csv', help="Path to the input dataset file")
#         parser.add_argument("--bucket", type=int, default=10, help="Size of the buckets for numerical values")
#         parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon value for differential privacy")

#         args = parser.parse_args()

#         data_path = args.dataset
#         bucket = args.bucket
#         epsilon = args.epsilon

#         data_original = pd.read_csv(data_path, engine='python')
#         data, bucketized_columns, removed_columns = preprocess_dataset(data_original, bucket)
#         print("\n[+] Head of the 'original' preprocessed dataset is: ")
#         print(data.head())
#         rows = data.shape[0]
#         print("\n[+] Number of rows in the dataset is: ", rows)

#         print("\n[+] Attribute Parent Pairs are being developed ...\n")
#         pb = PrivBayes(epsilon, n_cpus=6, score_function='R', verbose=True)
#         pb.fit(data)

#         starttime = time.time()
#         # Concatenate the results after all processes are completed
#         df_synth_original = pb.sample()
#         endtime = time.time()
#         print(f"[+] Time taken to generate {rows} records' synthetic dataset is: {endtime - starttime}\n\n")
#         df_synth = postprocess_dataset(df_synth_original, bucketized_columns, removed_columns)

#         print("\n[+] Head of the synthetic dataset is:\n")
#         print(df_synth.head())

#         marginal_comparison = MarginalComparison().fit(data_original, df_synth)
#         print("\n[+] Marginal comparision score is: ", marginal_comparison.score())
#         marginal_comparison.plot()
#         plt.savefig('Marginal_comparision.pdf')
#         print("-- Marginal Comparison Done & Graph is saved as a pdf file")

#         association_comparison = AssociationsComparison().fit(data_original, df_synth)
#         print("\n[+] Association comparision score is: ", association_comparison.score())
#         association_comparison.plot()
#         plt.savefig('Association_comparision.pdf')
#         print("-- Association Comparison Done & Graph is saved as a pdf file")
#         print("\n[+] Completed, Exiting...\n")

#     # Restore the original stdout
#     sys.stdout = original_stdout
#     # Print a message indicating that the output has been saved to a file
#     print("Output has been saved to Console-output.txt file")














## Below is the code with many debugging print statements..!!


# import numpy as np
# import pandas as pd
# import random
# import dill
# from sklearn.metrics import mutual_info_score
# from collections import namedtuple
# from joblib import Parallel, delayed
# import cProfile
# from concurrent.futures import ProcessPoolExecutor
# import matplotlib.pyplot as plt

# import synthesis.synthesizers.utils as utils
# from synthesis.synthesizers._base import BaseDPSynthesizer
# from thomas.core import BayesianNetwork
# from diffprivlib.mechanisms import Exponential
# from synthesis.evaluation.evaluator import SyntheticDataEvaluator
# from synthesis.evaluation.metrics import MarginalComparison, AssociationsComparison
# from synthesis.evaluation.efficacy import ClassifierComparison
# from synthesis.evaluation._base import BaseMetric
# from synthesis.evaluation.evaluator import DEFAULT_METRICS, SyntheticDataEvaluator

# APPair = namedtuple('APPair', ['attribute', 'parents'])




# class PrivBayes(BaseDPSynthesizer):
#     """
#     Implementation of the PrivBayes algorithm for differentially private data synthesis.
#     Inherits from the BaseDPSynthesizer class.
#     """

#     def __init__(self, epsilon=1.0, theta_usefulness=4, epsilon_split=0.3,
#                  score_function='R', network_init=None, n_cpus=None, verbose=True):
#         """
#         Initializes the PrivBayes synthesizer with the specified hyperparameters.
#         """
#         super().__init__(epsilon=epsilon, verbose=verbose)
#         self.theta_usefulness = theta_usefulness
#         self.epsilon_split = epsilon_split  # also called Beta in paper
#         self.score_function = score_function  # choose between 'R' and 'MI'
#         self.network_init = network_init
#         self.n_cpus = n_cpus

#     def fit(self, data):
#         """
#         Fits the PrivBayes synthesizer to the input data.
#         """
#         data = self._check_input_data(data)
#         self._check_init_args()

#         self._greedy_bayes(data)
#         self._compute_conditional_distributions(data)
#         self.model_ = BayesianNetwork.from_CPTs('PrivBayes', self.cpt_.values())
#         return self

#     def _check_init_args(self):
#         """
#         Checks the validity of the initialization arguments.
#         """
#         super()._check_init_args()
#         self._check_score_function()

#     def _check_score_function(self):
#         """
#         Checks the input score function and sets the sensitivity.
#         """
#         if self.score_function.upper() not in ['R', 'MI']:
#             raise ValueError("Score function must be 'R' or 'MI'")

#         if self.score_function.upper() == 'R':
#             self._score_sensitivity = (3 / self.n_records_fit_) + (2 / self.n_records_fit_**2)
#         elif self.score_function.upper() == 'MI':
#             self._score_sensitivity = (2 / self.n_records_fit_) * np.log((self.n_records_fit_ + 1) / 2) + \
#                               (((self.n_records_fit_ - 1) / self.n_records_fit_) *
#                                np.log((self.n_records_fit_ + 1) / (self.n_records_fit_ - 1)))

#     def sample(self, n_records=None):
#         """
#         Generates synthetic data using the PrivBayes synthesizer.
#         """
#         self._check_is_fitted()
#         n_records = n_records or self.n_records_fit_

#         data_synth = self._generate_data(n_records)
#         data_synth = self._check_output_data(data_synth)

#         if self.verbose:
#             print("\n[+] Synthetic Dataset is Generated\n")
#         return data_synth

#     def _greedy_bayes(self, data):
#         """
#         Performs the greedy Bayesian network construction algorithm.
#         """
#         nodes, nodes_selected = self._init_network(data)

#         # normally len(nodes) - 1, unless user initialized part of the network
#         self._n_nodes_dp_computed = len(nodes) - len(nodes_selected)

#         for i in range(len(nodes_selected), len(nodes)):
#             if self.verbose:
#                 print("[*] {}/{} - Evaluating next attribute to add to network".format(i + 1, len(self.columns_)))

#             nodes_remaining = nodes - nodes_selected

#             # select ap_pair candidates
#             ap_pairs = []
#             for node in nodes_remaining:
#                 max_domain_size = self._max_domain_size(data, node)
#                 max_parent_sets = self._max_parent_sets(data, nodes_selected, max_domain_size)

#                 # empty set - domain size of node violates theta_usefulness
#                 if len(max_parent_sets) == 0:
#                     ap_pairs.append(APPair(node, parents=None))
#                 # [empty set] - no parents found that meets domain size restrictions
#                 elif len(max_parent_sets) == 1 and len(max_parent_sets[0]) == 0:
#                     ap_pairs.append(APPair(node, parents=None))
#                 else:
#                     ap_pairs.extend([
#                         APPair(node, parents=tuple(p)) for p in max_parent_sets
#                     ])
#             if self.verbose:
#                 print("-- Number of AttributeParentPair candidates: {}".format(len(ap_pairs)))

#             scores = self._compute_scores(data, ap_pairs)
#             sampled_pair = self._exponential_mechanism(ap_pairs, scores)

#             if self.verbose:
#                 print("-- Selected attribute: '{}' - with parents: {}\n".format(sampled_pair.attribute, sampled_pair.parents))
#             nodes_selected.add(sampled_pair.attribute)
#             self.network_.append(sampled_pair)
#         if self.verbose:
#             print("\n[+] Learned Network Structure is being developed...")
#         return self

#     def _max_domain_size(self, data, node):
#         """Computes the maximum domain size a node can have to satisfy theta-usefulness"""
#         node_cardinality = utils.cardinality(data[node])
#         max_domain_size = (self.n_records_fit_ * (1 - self.epsilon_split) * self.epsilon) / \
#                           (2 * len(self.columns_) * self.theta_usefulness * node_cardinality)
#         return max_domain_size

#     def _max_parent_sets(self, data, v, max_domain_size):
#         """Refer to algorithm 5 in paper - max parent set is 1) theta-useful and 2) maximal."""
#         if max_domain_size < 1:
#             return set()
#         if len(v) == 0:
#             return [set()]

#         x = np.random.choice(tuple(v))
#         x_domain_size = utils.cardinality(data[x])
#         x = {x}

#         v_without_x = v - x

#         parent_sets1 = self._max_parent_sets(data, v_without_x, max_domain_size)
#         parent_sets2 = self._max_parent_sets(data, v_without_x, max_domain_size / x_domain_size)

#         for z in parent_sets2:
#             if z in parent_sets1:
#                 parent_sets1.remove(z)
#             parent_sets1.append(z.union(x))
#         return parent_sets1

#     def _init_network(self, X):
#         self._binary_columns = [c for c in X.columns if X[c].unique().size <= 2]
#         nodes = set(X.columns)

#         if self.network_init is not None:
#             nodes_selected = set(n.attribute for n in self.network_init)
#             print("Pre-defined network init: {}".format(self.network_))
#             for i, pair in enumerate(self.network_init):
#                 if self.verbose:
#                     print("{}/{} - init node {} - with parents: {}".format(i + 1, len(nodes),
#                                                                            pair.attribute, pair.parents))
#             self.network_ = self.network_init.copy()
#             return nodes, nodes_selected

#         # if set_network is not called we start with a random first node
#         self.network_ = []
#         nodes_selected = set()

#         root = np.random.choice(tuple(nodes))
#         self.network_.append(APPair(attribute=root, parents=None))
#         nodes_selected.add(root)
#         if self.verbose:
#             print("[*] 1/{} - Root of network: {}\n".format(X.shape[1], root))
#         return nodes, nodes_selected

#     def set_network(self, network):
#         assert [isinstance(n, APPair) for n in network], "input network does not consists of APPairs"
#         self.network_init = network
#         return self

#     def _compute_scores(self, data, ap_pairs):
#         """Compute score for all ap_pairs"""
#         if self.n_cpus:
#             scores = Parallel(n_jobs=self.n_cpus)(delayed(self.r_score)(data, pair.attribute, pair.parents) for pair in ap_pairs)
#         else:
#             scores = [self.r_score(data, pair.attribute, pair.parents) for pair in ap_pairs]
#         return scores

#     def _exponential_mechanism(self, ap_pairs, scores):
#         """select APPair with exponential mechanism"""
#         local_epsilon = self.epsilon * self.epsilon_split / self._n_nodes_dp_computed
#         #print("@@@@@@@@local_epsilon: ",local_epsilon)
#         dp_mech = Exponential(epsilon=local_epsilon, sensitivity=self._score_sensitivity,
#                               utility=list(scores), candidates=ap_pairs)
#         #print("@@@@@@@@dp_mech: ",dp_mech)
#         sampled_pair = dp_mech.randomise()
#         #print("@@@@@@@@sampled_pair: ",sampled_pair)
#         return sampled_pair
    
    
    
# #####    #! debug original _compute_conditional_distributions function
    
#     def _compute_conditional_distributions(self, data):
#             """
#             Compute the conditional distributions for each attribute in the network.
#             Uses differential privacy to ensure privacy guarantees.
            
#             Args:
#                 data: The input data used to compute the conditional distributions.
            
#             Returns:
#                 self: The updated object with computed conditional distributions.
#             """
#             self.cpt_ = dict()

#             local_epsilon = self.epsilon * (1 - self.epsilon_split) / len(self.columns_)
#             print("\n----START----[main code] The local_epsilon is: ",local_epsilon)
#             #print("\n[+]The Attribute-Parent Pairs developed are :\n",self.network_,"\n")
#             for idx, pair in enumerate(self.network_):
#                 if pair.parents is None:
#                     attributes = [pair.attribute]
#                 else:
#                     attributes = [*pair.parents, pair.attribute]

#                 cpt_size = utils.cardinality(data[attributes])
#                 if self.verbose:
#                     print('\n\n\n-- Learning conditional probabilities: {} - with parents {} '
#                         '~ estimated size: {}'.format(pair.attribute, pair.parents, cpt_size))

#                 dp_cpt = utils.dp_conditional_distribution(data[attributes], epsilon=local_epsilon)
#                 print("\n[main code] The dp_cpt is: ",dp_cpt)
#                 self.cpt_[pair.attribute] = dp_cpt
#             print("\n[main code]The self.cpt_ is: \n",self.cpt_)
#             #print("\n[main code]The entire self value is: ",self)
#             print("\n\n\n\n----END----[main code]\n\nPrinting the entire dictionory of the self\n\n",self.__dict__,"\n\n\n\n")

#             return self



#     # def _compute_conditional_distributions(self, data):

#     #     self.cpt_ = dict()

#     #     local_epsilon = self.epsilon * (1 - self.epsilon_split) / len(self.columns_)

#     #     for idx, pair in enumerate(self.network_):
#     #         if pair.parents is None:
#     #             attributes = [pair.attribute]
#     #         else:
#     #             attributes = [*pair.parents, pair.attribute]

#     #         # Discretize numerical attributes into buckets of size 10
#     #         for attr in attributes:
#     #             if pd.api.types.is_numeric_dtype(data[attr]):  # Check if the attribute is numerical
#     #                 data[attr] = pd.cut(data[attr], bins=10, labels=False, precision=1)

#     #                 print("\n[+] The numerical attribute is: ",attr)
#     #                 print("\n[+] The discretized attribute is: ",data[attr])
#     #                 print("\n[+] The discretized attribute type is: ",data[attr].dtype)
#     #                 print("\n[+] The discretized attribute unique values are: ",data[attr].unique())
#     #                 print("\n[+] The discretized attribute unique values count is: ",data[attr].unique().size)
#     #         cpt_size = utils.cardinality(data[attributes])
#     #         if self.verbose:
#     #             print('Learning conditional probabilities: {} - with parents {} '
#     #                 '~ estimated size: {}'.format(pair.attribute, pair.parents, cpt_size))

#     #         dp_cpt = utils.dp_conditional_distribution(data[attributes], epsilon=local_epsilon)
#     #         self.cpt_[pair.attribute] = dp_cpt

#     #     return self


#     def _generate_data(self, n_records = 30000):
#         """
#         Generate synthetic data based on the learned network and conditional distributions.
        
#         Args:
#             n_records: The number of records to generate (default: 30000).
        
#         Returns:
#             data_synth: The generated synthetic data.
#         """
#         data_synth = np.empty([n_records, len(self.columns_)], dtype=object)

#         for i in range(n_records):
#             # Generate a record by sampling from each attribute's conditional distribution
#             for j, pair in enumerate(self.network_):
#                 if pair.parents is None:
#                     attributes = [pair.attribute]
#                 else:
#                     attributes = [*pair.parents, pair.attribute]
                
#                 # Sample from the conditional distribution
#                 sample = utils.sample_from_distribution(self.cpt_[pair.attribute], data_synth[i, attributes])
                
#                 # Assign the sampled value to the corresponding attribute in the generated record
#                 data_synth[i, pair.attribute] = sample
        
#         return data_synth



# #### #! debug the below is the original code

#     def _generate_data(self, n_records = 30000):
#         data_synth = np.empty([n_records, len(self.columns_)], dtype=object)

#         for i in range(n_records):
#             if self.verbose:
#                 print("\r", end='')
#                 print('Number of records generated: {} / {}'.format(i + 1, n_records), end='', flush=True)
#             record = self._sample_record()
#             data_synth[i] = list(record.values())
#             # numpy.array to pandas.DataFrame with original column ordering
#         data_synth = pd.DataFrame(data_synth, columns=[c.attribute for c in self.network_])
#         return data_synth
# ###### till here is the original code


    


# ##### #! debug till here is the updated code..!


#     # def _sample_record(self):
#     #     """samples a value column for column by conditioning for parents"""
#     #     record = {}
#     #     for col_idx, pair in enumerate(self.network_):
#     #         node = self.model_[pair.attribute]
#     #         node_cpt = node.cpt
#     #         node_states = node.states

#     #         if node.conditioning:
#     #             parent_values = [record[p] for p in node.conditioning]
#     #             node_probs = node_cpt[tuple(parent_values)]

#     #         else:
#     #             node_probs = node_cpt.values
#     #         # use random.choices over np.random.choice as np coerces, e.g. sample['nan', 1, 3.0] -> '1' (int to string)
#     #         sampled_node_value = random.choices(node_states, weights=node_probs, k=1)[0] # returns list

#     #         record[node.name] = sampled_node_value
#     #     return record
    
    
#     """Below is the main optimized version which shortens timespan by almost 15 secs for 30000 records"""


#     import numpy as np

#     def _sample_record(self):
#         record = {}
#         for col_idx, pair in enumerate(self.network_):
#             node = self.model_[pair.attribute]
#             node_cpt = node.cpt
#             node_states = node.states

#             if node.conditioning:
#                 parent_values = [record[p] for p in node.conditioning]
#                 node_probs = node_cpt[tuple(parent_values)]
#             else:
#                 node_probs = node_cpt.values

#             # Use numpy.random.choice for faster random sampling
#             sampled_node_value = np.random.choice(node_states, p=node_probs)
#             record[node.name] = sampled_node_value

#         return record
        
        
# #### 10 sec optimization 

#     """def _sample_record(self):
#         record = {}
#         for col_idx, pair in enumerate(self.network_):
#             node = self.model_[pair.attribute]
#             node_cpt = node.cpt
#             node_states = node.states

#             if node.conditioning:
#                 parent_values = [record[p] for p in node.conditioning]
#                 node_probs = node_cpt[tuple(parent_values)]

#             else:
#                 node_probs = node_cpt.values

#             # Use NumPy for weighted random sampling
#             sampled_node_value = np.random.choice(node_states, p=node_probs)

#             record[node.name] = sampled_node_value
#         #print(record)
#         return record"""


# #### 10 sec optimization
#     """def _sample_record(self):

#         record = {}
#         for col_idx, pair in enumerate(self.network_):
#             node = self.model_[pair.attribute]
#             node_cpt = node.cpt
#             node_states = node.states

#             # Pre-calculate parent values for conditional nodes
#             if node.conditioning:
#                 parent_values = tuple([record[p] for p in node.conditioning])
#             else:
#                 parent_values = None

#             # Access conditional probabilities efficiently
#             node_probs = node_cpt[parent_values] if parent_values is not None else node_cpt.values

#             # Leverage NumPy for optimized sampling
#             sampled_node_value = np.random.choice(node_states, p=node_probs)
#             record[node.name] = sampled_node_value

#         return record"""
        
        
# #### The code below uses caches mechanism, it reduces time to less than 10 seconds, but the error is too much, and the synthetic data records are not as expected.       
#     """import numpy as np
#     from functools import lru_cache  # For caching

#     @lru_cache(maxsize=None)  # Cache function results for unchanged inputs
#     def _sample_record(self):

#         record = {}
#         for col_idx, pair in enumerate(self.network_):
#             node = self.model_[pair.attribute]
#             node_cpt = node.cpt
#             node_states = node.states

#             # Pre-calculate parent values (potentially vectorized)
#             if node.conditioning:
#                 parent_values = tuple([record[p] for p in node.conditioning])
#                 # Explore vectorizing parent_value calculations if applicable
#             else:
#                 parent_values = None

#             # Access conditional probabilities efficiently
#             node_probs = node_cpt[parent_values] if parent_values is not None else node_cpt.values

#             # Leverage NumPy for optimized sampling
#             sampled_node_value = np.random.choice(node_states, p=node_probs)
#             record[node.name] = sampled_node_value

#         return record"""
        

    


#     @staticmethod
#     def mi_score(data, columns_a, columns_b):
#         columns_a = utils._ensure_arg_is_list(columns_a)
#         columns_b = utils._ensure_arg_is_list(columns_b)

#         data_a = data[columns_a].squeeze()
#         if len(columns_b) == 1:
#             data_b = data[columns_b].squeeze()
#         else:
#             data_b = data.loc[:, columns_b].apply(lambda x: ' '.join(x.values), axis=1).squeeze()
#         return mutual_info_score(data_a, data_b)

#     @staticmethod
#     def mi_score_thomas(data, columns_a, columns_b):
#         columns_a = utils._ensure_arg_is_list(columns_a)
#         columns_b = utils._ensure_arg_is_list(columns_b)

#         prob_a = utils.compute_distribution(data[columns_a])
#         prob_b = utils.compute_distribution(data[columns_b])
#         prob_joint = utils.compute_distribution(data[columns_a + columns_b])

#         # todo: pull-request thomas to add option to normalize to remove 0's
#         # align
#         prob_div = prob_joint / (prob_b * prob_a)
#         prob_joint, prob_div = prob_joint.extend_and_reorder(prob_joint, prob_div)

#         # remove zeros as this will result in issues with log
#         prob_joint = prob_joint.values[prob_joint.values != 0]
#         prob_div = prob_div.values[prob_div.values != 0]
#         mi = np.sum(prob_joint * np.log(prob_div))
#         # mi = np.sum(p_nodeparents.values * np.log(p_nodeparents / (p_parents * p_node)))
#         return mi


#     @staticmethod
#     def r_score(data, columns_a, columns_b):
#         """An alternative score function to mutual information with lower sensitivity - can be used on non-binary domains.
#         Relies on the L1 distance from a joint distribution to a joint distributions that minimizes mutual information.
#         Refer to Lemma 5.2
#         """
#         if columns_b is None:
#             return 0
#         columns_a = utils._ensure_arg_is_list(columns_a)
#         columns_b = utils._ensure_arg_is_list(columns_b)

#         # compute distribution that minimizes mutual information
#         prob_a = utils.compute_distribution(data[columns_a])
#         prob_b = utils.compute_distribution(data[columns_b])
#         prob_independent = prob_b * prob_a

#         # compute joint distribution
#         prob_joint = utils.joint_distribution(data[columns_a + columns_b])

#         # substract not part of thomas - need to ensure alignment
#         # todo: should be part of thomas - submit pull-request to thomas
#         prob_joint, prob_independent = prob_joint.extend_and_reorder(prob_joint, prob_independent)
#         l1_distance = 0.5 * np.sum(np.abs(prob_joint.values - prob_independent.values))
#         return l1_distance

#     def save(self, path):
#         """
#         Save this synthesizer instance to the given path using pickle.

#         Parameters
#         ----------
#         path: str
#             Path where the synthesizer instance is saved.
#         """
#         # todo issue can't save if model is fitted - likely error within thomas
#         if hasattr(self, 'model_'):
#             pb = self.copy()
#             del pb.model_

#         with open(path, 'wb') as output:
#             dill.dump(pb, output)

#     @classmethod
#     def load(cls, path):
#         """Load a synthesizer instance from specified path.
#         Parameters
#         ----------
#         path: str
#             Path where the synthesizer instance is saved.

#         Returns
#         -------
#         synthesizer : class
#             Returns synthesizer instance.
#         """
#         with open(path, 'rb') as f:
#             pb = dill.load(f)

#         # recreate model_ attribute based on fitted network and cpt's
#         if hasattr(pb, 'cpt_'):
#             pb.model_ = BayesianNetwork.from_CPTs('PrivBayes', pb.cpt_.values())
#         return pb



# class PrivBayesFix(PrivBayes):
#     """Extension to PrivBayes class to allow user to fix pre-sampled columns.
#     Can be used to generate additional items for an already released synthetic dataset.
#     """

#     def __init__(self, epsilon=1.0, theta_usefulness=4, epsilon_split=0.3,
#                  score_function='R', network_init=None, n_cpus=None, verbose=True):
#         super().__init__(epsilon=epsilon, theta_usefulness=theta_usefulness, epsilon_split=epsilon_split,
#                          score_function=score_function, network_init=network_init, n_cpus=n_cpus, verbose=verbose)

#         if network_init is not None:
#             self.network_init = network_init
#             self._init_network()

#     # def _init_network(self, X):
#     #     nodes = set(X.columns)

#     #     if self.network_init is not None:
#     #         nodes_selected = set(n.attribute for n in self.network_init)
#     #         print("Pre-defined network init: {}".format(self.network_))
#     #         for i, pair in enumerate(self.network_init):
#     #             if self.verbose:
#     #                 print("{}/{} - init node {} - with parents: {}".format(i + 1, len(nodes),
#     #                                                                     pair.attribute, pair.parents))
#     #         self.network_ = self.network_init.copy()
#     #         return nodes, nodes_selected

#     #     # if set_network is not called, we start with a random first node
#     #     self.network_ = []
#     #     nodes_selected = set()

#     #     root = np.random.choice(tuple(nodes))
#     #     self.network_.append(APPair(attribute=root, parents=None))
#     #     nodes_selected.add(root)
#     #     if self.verbose:
#     #         print("1/{} - Root of network: {}\n".format(X.shape[1], root))
#     #     return nodes, nodes_selected
    
    
#     def _init_network(self, X):
#         nodes = set(X.columns)

#         if self.network_init is not None:
#             nodes_selected = set(n.attribute for n in self.network_init)
#             if self.verbose:
#                 print("Pre-defined network init: {}".format(self.network_init))
#             self.network_ = self.network_init.copy()
#             return nodes, nodes_selected

#         # If set_network is not called, we start with a random first node
#         self.network_ = []
#         nodes_selected = set()

#         root = np.random.choice(tuple(nodes))
#         self.network_.append(APPair(attribute=root, parents=None))
#         nodes_selected.add(root)
#         if self.verbose:
#             print("1/{} - Root of network: {}\n".format(X.shape[1], root))
#         return nodes, nodes_selected


#     def fit(self, data):
#         super().fit(data)
#         return self

#     def sample_remaining_columns(self, fixed_data):
#         self._check_is_fitted()
#         fixed_data = self._check_fixed_data(fixed_data)

#         data_synth = self._generate_data(fixed_data)
#         data_synth = self._check_output_data(data_synth)

#         if self.verbose:
#             print("\nRemaining synthetic columns generated")
#         return data_synth

#     def _generate_data(self, fixed_data):
#         n_records = fixed_data.shape[0]
#         data_synth = np.empty([n_records, len(self.columns_)], dtype=object)

#         for i in range(n_records):
#             if self.verbose:
#                 print("\r", end='')
#                 print('Number of records generated: {} / {}'.format(i + 1, n_records), end='', flush=True)
#             print("\n")

#             record_init = fixed_data.loc[i].to_dict()
#             record = self._sample_record(record_init)
#             data_synth[i] = list(record.values())

#         # numpy.array to pandas.DataFrame with original column ordering
#         data_synth = pd.DataFrame(data_synth, columns=[c.attribute for c in self.network_])
#         return data_synth

#     # def _sample_record(self, record_init):
#     #     # assume X has columns with values that correspond to the first nodes in the network
#     #     # that we would like to fix and condition for.
#     #     record = record_init

#     #     # sample remaining nodes after fixing for input X
#     #     for col_idx, pair in enumerate(self.network_[len(record_init):]):
#     #         node = self.model_[pair.attribute]

#     #         # specify pre-sampled conditioning values
#     #         node_cpt = node.cpt
#     #         node_states = node.states

#     #         if node.conditioning:
#     #             parent_values = [record[p] for p in node.conditioning]
#     #             node_probs = node_cpt[tuple(parent_values)]
#     #         else:
#     #             node_probs = node_cpt.values
#     #         # sampled_node_value = np.random.choice(node_states, p=node_probs)
#     #         sampled_node_value = random.choices(node_states, weights=node_probs)
#     #         record[node.name] = sampled_node_value
#     #     return record
    
    
    
#     def _sample_record(self, record_init):
#         record = record_init.copy()

#         # sample remaining nodes after fixing for input X
#         for col_idx, pair in enumerate(self.network_[len(record_init):]):
#             node = self.model_[pair.attribute]

#             # specify pre-sampled conditioning values
#             node_cpt = node.cpt
#             node_states = node.states

#             if node.conditioning:
#                 parent_values = [record[p] for p in node.conditioning]
#                 node_probs = node_cpt[tuple(parent_values)]
#             else:
#                 node_probs = node_cpt.values

#             # Use np.random.choice to avoid ValueError when weights don't sum to 1
#             sampled_node_value = np.random.choice(node_states, p=node_probs)
#             record[node.name] = sampled_node_value.item()  # Convert to scalar

#         return record

#     def _check_fixed_data(self, data):
#         """Checks whether the columns in fixed data where also seen in fit"""
#         if isinstance(data, pd.Series):
#             data = data.to_frame()

#         if not np.all([c in self.columns_ for c in data.columns]):
#             raise ValueError('Columns in fixed data not seen in fit.')
#         if set(data.columns) == set(self.columns_):
#             raise ValueError('Fixed data already contains all the columns that were seen in fit.')

#         if self.verbose:
#             sample_columns = [c for c in self.columns_ if c not in data.columns]
#             print('Columns sampled and added to fixed data: {}'.format(sample_columns))
#         # prevent integer column indexing issues
#         data.columns = data.columns.astype(str)
#         # make all columns categorical.
#         data = data.astype('category')
#         return data

# class PrivBayesNP(PrivBayes):
#     """Privbayes class with infinite-differential privacy, while still using epsilon value to limit the size of the network
#     """

#     def __init__(self, epsilon=1, theta_usefulness=4, epsilon_split=0.5,
#                  score_function='R', network_init=None, n_cpus=None, verbose=True):
#         self.epsilon1 = float(np.inf)
#         self.epsilon2 = epsilon
#         super().__init__(epsilon=self.epsilon1, theta_usefulness=theta_usefulness, epsilon_split=epsilon_split,
#                          score_function=score_function, network_init=network_init, n_cpus=n_cpus, verbose=verbose)


#     def _max_domain_size(self, data, node):
#         """Computes the maximum domain size a node can have to satisfy theta-usefulness"""
#         node_cardinality = utils.cardinality(data[node])
#         max_domain_size = self.n_records_fit_ * self.epsilon2 / \
#                           (2 * len(self.columns_) * self.theta_usefulness * node_cardinality)
#         return max_domain_size

# import time

# # def preprocess_dataset(input_data):
# #     print("\n[+] Preprocessing the dataset to convert numerical values into buckets of size 10 :")
# #     # Remove rows with missing values
# #     input_data = input_data.dropna()

# #     processed_data = input_data.copy()
# #     bucketized_columns = []

# #     for column in input_data.columns:
# #         if pd.api.types.is_numeric_dtype(input_data[column]):
# #             # Process numerical columns into buckets of size 10
# #             bucketized_values = pd.cut(input_data[column], bins=10, labels=False, precision=1)
# #             # Replace original numerical values with bucketized values in the same column
# #             processed_data[column] = bucketized_values
# #             # Print information about bucketized column
# #             print(f"\n[+] {column} is numerical. Bucketized values:")
# #             #print(bucketized_values.value_counts().sort_index())
# #             print(f"-- Domain size for {column}: {len(bucketized_values.unique())}")
# #             # Add the column to the list of bucketized columns
# #             bucketized_columns.append(column)
        
# #     return processed_data, bucketized_columns


# import pandas as pd
# import numpy as np

# def preprocess_dataset(input_data, bins):
#     print(f"\n[+] Preprocessing the dataset to convert numerical values into buckets of size {bins}:")
#     # Remove rows with missing values
#     input_data = input_data.dropna()

#     bucket_mappings = {}  # Dictionary to store mapping between original values and bucketized values

#     for column in input_data.columns:
#         if pd.api.types.is_numeric_dtype(input_data[column]):
#             # Process numerical columns into buckets of size 10
#             bucketized_values, mapping = pd.cut(input_data[column], bins, labels=False, precision=0, retbins=True)
#             # Replace original numerical values with bucketized values in the same column
#             input_data[column] = bucketized_values.astype(int)
#             bucket_mappings[column] = mapping.astype(int)
#             # Print information about bucketized column
#             print(f"\n[+] {column} is numerical. Bucketized values:")
#             #print(bucketized_values.value_counts().sort_index())
#             print(f"-- Domain size for {column}: {len(bucketized_values.unique())}")

#     return input_data, bucket_mappings


# # import numpy as np
# # import pandas as pd

# # def preprocess_dataset(input_data):
# #     print("\n[+] Preprocessing the dataset...")

# #     # Remove rows with missing values
# #     input_data = input_data.dropna()

# #     bucket_mappings = {}

# #     for column in input_data.columns:
# #         if pd.api.types.is_integer_dtype(input_data[column]):
# #             num_unique = len(input_data[column].unique())
# #             print(f"\n[+] {column} is numerical. Number of unique values: {num_unique}")

# #             if num_unique < 50:  # Treat as categorical
# #                 print(f"-- Treating {column} as categorical due to low cardinality.")
# #                 input_data[column] = pd.Categorical(input_data[column])
# #                 bucket_mappings[column] = input_data[column].cat.categories
# #             else:
# #                 print(f"-- Bucketizing {column} with adaptive bin size.")

# #                 # Handle non-finite values before bucketizing
# #                 input_data[column].replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
# #                 input_data.dropna(subset=[column], inplace=True)  # Drop rows with NaN in the column

# #                 # Determine bin size based on unique values and default numeric_bin_size (5)
# #                 numeric_bin_size = 5
# #                 bin_size = min(numeric_bin_size, num_unique)

# #                 # Generate equal-width bins
# #                 lowest = input_data[column].min()
# #                 highest = input_data[column].max()
# #                 bins = np.linspace(lowest, highest, bin_size + 1)

# #                 # Categorize numerical values into bins
# #                 bucketized_values = pd.cut(input_data[column], bins, labels=False) + 1  # Start indices from 1
# #                 input_data[column] = bucketized_values.astype(int)

# #                 # Store mapping (bin edges)
# #                 bucket_mappings[column] = bins.astype(int)

# #     return input_data, bucket_mappings


# def postprocess_dataset(data, bucket_mappings):
#     converted_data = data.copy()

#     for column, mapping in bucket_mappings.items():
#         if column in converted_data.columns:
#             # Replace bucketized values with the average of neighboring array elements
#             bucketized_values = converted_data[column]
#             original_values = np.zeros_like(bucketized_values)

#             for i, value in enumerate(bucketized_values):
#                 if np.isfinite(value):
#                     lower_index = int(value) - 1  # Adjust for zero-based indexing
#                     upper_index = int(value)
#                     average = (mapping[lower_index] + mapping[upper_index]) / 2
#                     original_values[i] = int(average)  # Convert to integer directly
#                 else:
#                     original_values[i] = -1  # Default for non-finite values

#             converted_data[column] = original_values

#     return converted_data

# import argparse

# if __name__ == '__main__':
    
#     # """Main function to demonstrate the usage of PrivBayes and PrivBayesFix."""
#     # data_path = '../input-data/adult.csv'
#     # data_original = pd.read_csv(data_path, engine='python')
#     # # columns = ['age', 'sex', 'education', 'workclass', 'income']
#     # # data = data.loc[:, columns]
    

#     parser = argparse.ArgumentParser(description="Optimized PrivBayes main function")
#     parser.add_argument("--dataset", type=str, default = 'input-data/adult_tiny.csv' ,help="Path to the input dataset file")
#     parser.add_argument("--bucket", type=int, default=10, help="Size of the buckets for numerical values")
#     parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon value for differential privacy")
    
    
#     args = parser.parse_args()

#     data_path = args.dataset
#     bucket = args.bucket
#     epsilon = args.epsilon
    
#     data_original = pd.read_csv(data_path, engine='python')   
#     data,bucketized_columns = preprocess_dataset(data_original, bucket)
#     print("\n[+] Head of the 'original' preprocessed dataset is: ")
#     print(data.head())
#     rows = data.shape[0]
#     print("\n[+] Number of rows in the dataset is: ",rows)
    
#     print("\n[+] Attribute Parent Pairs are being developed ...\n")
#     pb = PrivBayes(epsilon, n_cpus=6, score_function='R', verbose=True)
#     pb.fit(data)

#     starttime = time.time()
#     # Concatenate the results after all processes are completed
#     df_synth_original = pb.sample()
#     endtime = time.time()
#     print(f"[+] Time taken to generate {rows} records' synthetic dataset is: {endtime - starttime}\n")
#     df_synth = postprocess_dataset(df_synth_original, bucketized_columns)
    
#     print("\n[+] Head of the synthetic dataset is:\n")
#     print(df_synth.head())
#     # result4 = pb.score(data, df_synth, score_dict=True)
#     # print("\n[+] Score of the synthetic dataset is: \n",result4)

#     # """test scoring functions"""
#     # pair = pb.network_[3]
#     # result1 = pb.mi_score(data, pair.attribute, pair.parents)
#     # print("\n[+] Mutual Information Score: ", result1)
#     # result2 = pb.mi_score_thomas(data, pair.attribute, pair.parents)
#     # print("\n[+] Mutual Information Score Thomas: ", result2)
#     # result3 = pb.r_score(data, pair.attribute, pair.parents)
#     # print("\n[+] R Score: ", result3)
    
    
    
#     marginal_comparison= MarginalComparison().fit(data, df_synth_original)
#     print("\n[+] Marginal comparision score is: ",marginal_comparison.score())
#     marginal_comparison.plot()
#     plt.savefig('Marginal_comparision.pdf')
#     #plt.show()
#     #plt.close()
#     print("-- Marginal Comparison Done & Graph is saved as a pdf file")
    
    
#     association_comparison= AssociationsComparison().fit(data, df_synth_original)
#     print("\n[+] Association comparision score is: ",association_comparison.score())
#     association_comparison.plot()
#     plt.savefig('Association_comparision.pdf')
#     #plt.show()
#     #plt.close()  
#     print("-- Association Comparison Done & Graph is saved as a pdf file")
#     print("\n[+] Completed, Exiting...\n")  







