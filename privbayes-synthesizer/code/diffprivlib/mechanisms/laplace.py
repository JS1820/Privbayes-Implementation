# MIT License
#
# Copyright (C) IBM Corporation 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
The classic Laplace mechanism in differential privacy, and its derivatives.
"""
from numbers import Real

import numpy as np

#from .laplace import nonnegativity
from diffprivlib.mechanisms.base import DPMechanism, TruncationAndFoldingMixin
from diffprivlib.utils import copy_docstring


class Laplace(DPMechanism):
    r"""
    The classical Laplace mechanism in differential privacy.

    First proposed by Dwork, McSherry, Nissim and Smith [DMNS16]_, with support for (relaxed)
    :math:`(\epsilon,\delta)`-differential privacy [HLM15]_.

    Samples from the Laplace distribution are generated using 4 uniform variates, as detailed in [HB21]_, to prevent
    against reconstruction attacks due to limited floating point precision.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    References
    ----------
    .. [DMNS16] Dwork, Cynthia, Frank McSherry, Kobbi Nissim, and Adam Smith. "Calibrating noise to sensitivity in
        private data analysis." Journal of Privacy and Confidentiality 7, no. 3 (2016): 17-51.

    .. [HLM15] Holohan, Naoise, Douglas J. Leith, and Oliver Mason. "Differential privacy in metric spaces: Numerical,
        categorical and functional data under the one roof." Information Sciences 305 (2015): 256-268.

    .. [HB21] Holohan, Naoise, and Stefano Braghin. "Secure Random Sampling in Differential Privacy." arXiv preprint
        arXiv:2107.10138 (2021).

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, random_state=random_state)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = None

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        return float(sensitivity)

    def _check_all(self, value):
        super()._check_all(value)
        self._check_sensitivity(self.sensitivity)

        if not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")

        return True

    def bias(self, value):
        """Returns the bias of the mechanism at a given `value`.

        Parameters
        ----------
        value : int or float
            The value at which the bias of the mechanism is sought.

        Returns
        -------
        bias : float or None
            The bias of the mechanism at `value`.

        """
        return 0.0

    def variance(self, value):
        """Returns the variance of the mechanism at a given `value`.

        Parameters
        ----------
        value : float
            The value at which the variance of the mechanism is sought.

        Returns
        -------
        bias : float
            The variance of the mechanism at `value`.

        """
        self._check_all(0)

        return 2 * (self.sensitivity / (self.epsilon - np.log(1 - self.delta))) ** 2

    @staticmethod
    def _laplace_sampler(unif1, unif2, unif3, unif4):
        return np.log(1 - unif1) * np.cos(np.pi * unif2) + np.log(1 - unif3) * np.cos(np.pi * unif4)

    # def randomise(self, value):
    #     """Randomise `value` with the mechanism.

    #     Parameters
    #     ----------
    #     value : float
    #         The value to be randomised.

    #     Returns
    #     -------
    #     float
    #         The randomised value.

    #     """
    #     self._check_all(value)
    #     print('[super.randomize] Inside the laplace class randomize, and Value: {}'.format(value))

    #     scale = self.sensitivity / (self.epsilon - np.log(1 - self.delta))
    #     print('[super.randomize] Inside the laplace class randomize, and Scale: {}'.format(scale))
    #     standard_laplace = self._laplace_sampler(self._rng.random(), self._rng.random(), self._rng.random(),
    #                                              self._rng.random())
    #     print('[super.randomize] Inside the laplace class randomize, and Standard Laplace: {}'.format(standard_laplace))
    #     print('[super.randomize] returns the new randomized value as: {}'.format(value - scale * standard_laplace))
    #     noisy_value = value - scale * standard_laplace
    #     return value - scale * standard_laplace

    def nonnegativity(self, array, rho=0.0001, T=30):
        """Enforces non-negativity on an array of integers, gradually reducing negative values.

        Args:
            array: The input array of integers.
            rho: The threshold for absolute sum of negative numbers (default: 0.0001).
            T: The maximum number of rounds to perform (default: 30).

        Returns:
            The modified array with non-negative values.
        """

        round_count = 0
        negative_sum = 0
        print("\n\n\n\n\n\n[+START][non-negativity] Initial array before non-negativity enforcement is:\n\n")
        print(array)
        while round_count < T and abs(sum(x for x in array if x < 0)) > rho:
            round_count += 1
            print(f"\n\n          ==============================[non-negativity] Inside the non-negativity enforcement loop. Round count is {round_count} :====================================\n")
            for i in range(len(array)):
                if array[i] < 0:
                    print(f"\n     [ROUND {round_count}] [non-negativity] --- negative value found at position {i}:", array[i])
                else:
                    print(f"\n     [ROUND {round_count}] [non-negativity] +++ positive value found at position {i}:", array[i])
            print("\n\n          ")
            negative_sum = sum(abs(x) for x in array if x < 0)
            print(f"\n     [ROUND {round_count}] [non-negativity] Absolute Sum of negative numbers above is:", negative_sum)
            positive_count = sum(1 for x in array if x > 0)  # Count positive values using generator expression
            print(f"\n     [ROUND {round_count}] [non-negativity] Count of positive numbers above is:", positive_count)
            
            try:
                height = negative_sum / positive_count  # Handle potential division by zero
            except ZeroDivisionError:
                height = 0

            print(f"\n     [ROUND {round_count}] [non-negativity] Height for round no.{round_count} is:", height)
            print("\n\n          ")
            for i in range(len(array)):
                if array[i] > 0:
                    print(f"\n     [ROUND {round_count}] [non-negativity] +++ positive value at position {i}:", array[i])
                    array[i] -= height
                    print(f"\n     [ROUND {round_count}] [non-negativity] +++ new value at position {i} after subtraction of height :", array[i])
                elif array[i] < 0:
                    print(f"\n     [ROUND {round_count}] [non-negativity] --- negative value at position {i}:", array[i])
                    array[i] = 0
                    print(f"\n     [ROUND {round_count}] [non-negativity] --- new value at position {i} after setting to 0 :", array[i])
        
        print("\n          ")
        
        
        # if round_count >= T:
        #     print(f"\n\n          [non-negativity] Number of rounds completed is {round_count}, And it's more than {T} !\n")
        # elif abs(sum(x for x in array if x < 0)) <= rho:
        #     print(f"\n\n          [non-negativity] Absolute sum of negative numbers {abs(sum(x for x in array if x < 0))} is below the threshold {rho} !\n")
        
        
        
        # print(f"\n          [non-negativity] Number of rounds completed is {round_count}, And Threshold is {T} !")
        # print(f"\n          [non-negativity] Absolute sum of negative numbers {negative_sum}, And Threshold is {rho} !")
        # print(f"\n          [non-negativity] Count of negative numbers {sum(1 for x in array if x < 0)} !")
        
        if sum(1 for x in array if x < 0) == 0:
            print(f"\n     [-QUIT][non-negativity] No more negative numbers found in the array !")
        elif abs(sum(x for x in array if x < 0)) < rho:
            print(f"\n     [-QUIT][non-negativity] Absolute sum of negative numbers {abs(sum(x for x in array if x < 0))} is below the threshold {rho} !")
        elif round_count >= T:
            print(f"\n     [-QUIT][non-negativity] Number of rounds completed is {round_count}, And its more than {T} !")
        
        
        
        
        
        print("\n     [-QUIT][non-negativity] Because of the above reason, we are setting any of the remaining negative values to 0")

        # Set any remaining negative values to 0
        for i in range(len(array)):
            if array[i] < 0:
                print(f"\n     [-REPLACE][non-negativity] Previous value at {i}, {array[i]} is replaced with 0")
                array[i] = 0
                # print(f"\n          [non-negativity] --- new value at position {i} after setting to 0 :", array[i])

        print("\n[-END][non-negativity] Final array after non-negativity enforcement is:\n\n")
        print(array)
        print("\n\n\n\n\n\n")
        return array


    def randomise(self, values):
        """Randomise `values` with the mechanism.

        Parameters
        ----------
        values : list or array-like
            The values to be randomised.

        Returns
        -------
        array-like
            The array of randomised values.

        """
        result_values = []
        for value in values:
            self._check_all(value)
        
        #print('[super.randomize] Inside the laplace class randomize, and Values: {}'.format(values))
        
        for value in values:
            scale = self.sensitivity / (self.epsilon - np.log(1 - self.delta))
            #print('[super.randomize] Inside the laplace class randomize, and Scale: {}'.format(scale))
            standard_laplace = self._laplace_sampler(self._rng.random(), self._rng.random(), self._rng.random(),
                                                    self._rng.random())
            #print('[super.randomize] Inside the laplace class randomize, and Standard Laplace: {}'.format(standard_laplace))
            noisy_value = value - scale * standard_laplace
            #print('[super.randomize] returns the new randomized value as: {}'.format(noisy_value))
            result_values.append(noisy_value)
            
        #print('\n[super.randomize] returns the new randomized values as: {}'.format(result_values))
        #return result_values
        #print("\n[super.randomize] The noisy values are sent for non-negativity enforcing..!")
        noisy_values = self.nonnegativity(result_values)
        return noisy_values


    
    
    
    # def randomise(self, values, threshold=0.3, T=30):
    #     """Randomise a list of values with the mechanism while enforcing non-negativity.

    #     Parameters
    #     ----------
    #     values : list or array-like
    #         The values to be randomised.
    #     threshold : float, optional
    #         The threshold for stopping the non-negativity loop. Default is 0.3.
    #     T : int, optional
    #         The maximum number of rounds for non-negativity. Default is 30.

    #     Returns
    #     -------
    #     array-like
    #         The array of randomised values.

    #     """
    #     # Check all values in the list
    #     for value in values:
    #         self._check_all(value)

    #     print('[super.randomize] Inside the laplace class randomize, and Values: {}'.format(values))

    #     scale = self.sensitivity / (self.epsilon - np.log(1 - self.delta))
    #     print('[super.randomize] Inside the laplace class randomize, and Scale: {}'.format(scale))

    #     sum_negative = 0  # Variable to store the sum of negative numbers
    #     num_positive = 0  # Variable to count the number of positive numbers
    #     T_counter = 0  # Variable to count the number of rounds of non-negativity

    #     while True:
    #         randomized_values = []  # List to store the randomised values

    #         for value in values:
    #             standard_laplace = self._laplace_sampler(self._rng.random(), self._rng.random(), self._rng.random(),
    #                                                     self._rng.random())
    #             randomized_value = value - scale * standard_laplace

    #             if randomized_value < 0:
    #                 sum_negative += abs(randomized_value)
    #                 num_positive += 1
    #                 randomized_value = 0  # Set negative value to 0 for now

    #             randomized_values.append(randomized_value)

    #         if sum_negative / num_positive > threshold or T_counter >= T:
    #             return np.maximum(0, randomized_values)  # If threshold is exceeded or T rounds reached, return values with negative values set to 0
    #         else:
    #             print("[super.randomize] SUM_negative: {}, NUM_positive: {}".format(sum_negative, num_positive))
    #             values = np.maximum(0, randomized_values - sum_negative / num_positive)  # Adjust values for non-negativity
    #             sum_negative = 0  # Reset sum_negative for the next round
    #             num_positive = 0  # Reset num_positive for the next round
    #             T_counter += 1
    #         return randomized_values



class LaplaceTruncated(Laplace, TruncationAndFoldingMixin):
    r"""
    The truncated Laplace mechanism, where values outside a pre-described domain are mapped to the closest point
    within the domain.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        self._check_all(value)

        shape = self.sensitivity / self.epsilon
        # print('@@@@ Inside the laplace truncated class, and Shape: {}'.format(shape))
        # print('@@@@ Inside the laplace truncated class, and Value: {}'.format(value))
        # print('@@@@ Inside the laplace truncated class, and Lower: {}'.format(self.lower))
        # print('@@@@ Inside the laplace truncated class, and Upper: {}'.format(self.upper))
        # print('@@@@ Inside the laplace truncated class, and returns ----: {}',shape / 2 * (np.exp((self.lower - value) / shape) - np.exp((value - self.upper) / shape)))
        return shape / 2 * (np.exp((self.lower - value) / shape) - np.exp((value - self.upper) / shape))

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        self._check_all(value)

        shape = self.sensitivity / self.epsilon

        variance = value ** 2 + shape * (self.lower * np.exp((self.lower - value) / shape)
                                         - self.upper * np.exp((value - self.upper) / shape))
        variance += (shape ** 2) * (2 - np.exp((self.lower - value) / shape)
                                    - np.exp((value - self.upper) / shape))

        variance -= (self.bias(value) + value) ** 2

        return variance

    def _check_all(self, value):
        Laplace._check_all(self, value)
        TruncationAndFoldingMixin._check_all(self, value)

        return True

    @copy_docstring(Laplace.randomise)
    # def randomise(self, value):
    #     print('----START(randomize)---- Inside the laplace truncated randomize class, and original value is: {}'.format(value))
    #     self._check_all(value)
    #     #print('@@@@ Inside the laplace truncated randomize class, and Value: {}'.format(value))
    #     noisy_value = super().randomise(value)
    #     print('----END(randomize)---- The noisy value returned after randomizing the original value is: {}'.format(noisy_value))
    #     print('@@@@ Inside the laplace truncated randomize class, and returns-------------->>>>>>>: ',self._truncate(noisy_value))
    #     return self._truncate(noisy_value)
    
    
    def randomise(self, values):
        """Randomise a list of values with the mechanism.

        Parameters
        ----------
        values : list or array-like
            The values to be randomised.

        Returns
        -------
        array-like
            The array of randomised values.

        """
        # print('----START(randomize)---- Inside the laplace truncated randomize class, and original values are: {}'.format(values))
        
        # Ensure all values in the list pass the check
        for value in values:
            self._check_all(value)
        
        # Send the entire list of values for randomization to the superclass
        noisy_values = super().randomise(values)
        
        # print('----END(randomize)---- The noisy values returned after randomizing the original values are: {}'.format(noisy_values))
        #print('@@@@ Inside the laplace truncated randomize class, and returns-------------->>>>>>>: ', self._truncate(noisy_values))
        
        return noisy_values
        #return self._truncate(noisy_values)



class LaplaceFolded(Laplace, TruncationAndFoldingMixin):
    r"""
    The folded Laplace mechanism, where values outside a pre-described domain are folded around the domain until they
    fall within.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        self._check_all(value)

        shape = self.sensitivity / self.epsilon

        bias = shape * (np.exp((self.lower + self.upper - 2 * value) / shape) - 1)
        bias /= np.exp((self.lower - value) / shape) + np.exp((self.upper - value) / shape)

        return bias

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        raise NotImplementedError

    def _check_all(self, value):
        super()._check_all(value)
        TruncationAndFoldingMixin._check_all(self, value)

        return True

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        noisy_value = super().randomise(value)
        return self._fold(noisy_value)



class LaplaceBoundedDomain(LaplaceTruncated):
    r"""
    The bounded Laplace mechanism on a bounded domain.  The mechanism draws values directly from the domain using
    rejection sampling, without any post-processing [HABM20]_.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    References
    ----------
    .. [HABM20] Holohan, Naoise, Spiros Antonatos, Stefano Braghin, and Pól Mac Aonghusa. "The Bounded Laplace Mechanism
        in Differential Privacy." Journal of Privacy and Confidentiality 10, no. 1 (2020).

    """
    def _find_scale(self):
        eps = self.epsilon
        delta = self.delta
        diam = self.upper - self.lower
        delta_q = self.sensitivity

        def _delta_c(shape):
            if shape == 0:
                return 2.0
            return (2 - np.exp(- delta_q / shape) - np.exp(- (diam - delta_q) / shape)) / (1 - np.exp(- diam / shape))

        def _f(shape):
            return delta_q / (eps - np.log(_delta_c(shape)) - np.log(1 - delta))

        left = delta_q / (eps - np.log(1 - delta))
        right = _f(left)
        old_interval_size = (right - left) * 2

        while old_interval_size > right - left:
            old_interval_size = right - left
            middle = (right + left) / 2

            if _f(middle) >= middle:
                left = middle
            if _f(middle) <= middle:
                right = middle

        return (right + left) / 2

    def effective_epsilon(self):
        r"""Gets the effective epsilon of the mechanism, only for strict :math:`\epsilon`-differential privacy.  Returns
        ``None`` if :math:`\delta` is non-zero.

        Returns
        -------
        float
            The effective :math:`\epsilon` parameter of the mechanism.  Returns ``None`` if `delta` is non-zero.

        """
        if self._scale is None:
            self._scale = self._find_scale()

        if self.delta > 0.0:
            return None

        return self.sensitivity / self._scale

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        self._check_all(value)

        if self._scale is None:
            self._scale = self._find_scale()

        bias = (self._scale - self.lower + value) / 2 * np.exp((self.lower - value) / self._scale) \
            - (self._scale + self.upper - value) / 2 * np.exp((value - self.upper) / self._scale)
        bias /= 1 - np.exp((self.lower - value) / self._scale) / 2 \
            - np.exp((value - self.upper) / self._scale) / 2

        return bias

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        self._check_all(value)

        if self._scale is None:
            self._scale = self._find_scale()

        variance = value**2
        variance -= (np.exp((self.lower - value) / self._scale) * (self.lower ** 2)
                     + np.exp((value - self.upper) / self._scale) * (self.upper ** 2)) / 2
        variance += self._scale * (self.lower * np.exp((self.lower - value) / self._scale)
                                   - self.upper * np.exp((value - self.upper) / self._scale))
        variance += (self._scale ** 2) * (2 - np.exp((self.lower - value) / self._scale)
                                          - np.exp((value - self.upper) / self._scale))
        variance /= 1 - (np.exp(-(value - self.lower) / self._scale)
                         + np.exp(-(self.upper - value) / self._scale)) / 2

        variance -= (self.bias(value) + value) ** 2

        return variance

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        if self._scale is None:
            self._scale = self._find_scale()

        value = max(min(value, self.upper), self.lower)
        if np.isnan(value):
            return float("nan")

        samples = 1

        while True:
            try:
                unif = self._rng.random(4 * samples)
            except TypeError:  # rng is secrets.SystemRandom
                unif = [self._rng.random() for _ in range(4 * samples)]
            noisy = value + self._scale * self._laplace_sampler(*np.array(unif).reshape(4, -1))

            if ((noisy >= self.lower) & (noisy <= self.upper)).any():
                idx = np.argmax((noisy >= self.lower) & (noisy <= self.upper))
                return noisy[idx]
            samples = min(100000, samples * 2)


class LaplaceBoundedNoise(Laplace):
    r"""
    The Laplace mechanism with bounded noise, only applicable for approximate differential privacy (delta > 0)
    [GDGK18]_.

    Epsilon must be strictly positive, `epsilon` > 0. `delta` must be strictly in the interval (0, 0.5).
     - For zero `epsilon`, use :class:`.Uniform`.
     - For zero `delta`, use :class:`.Laplace`.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    delta : float
        Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 0.5).

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    random_state : int or RandomState, optional
        Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    References
    ----------
    .. [GDGK18] Geng, Quan, Wei Ding, Ruiqi Guo, and Sanjiv Kumar. "Truncated Laplacian Mechanism for Approximate
        Differential Privacy." arXiv preprint arXiv:1810.00877v1 (2018).

    """
    def __init__(self, *, epsilon, delta, sensitivity, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
        self._noise_bound = None

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if epsilon == 0:
            raise ValueError("Epsilon must be strictly positive. For zero epsilon, use :class:`.Uniform`.")

        if isinstance(delta, Real) and not 0 < delta < 0.5:
            raise ValueError("Delta must be strictly in the interval (0,0.5). For zero delta, use :class:`.Laplace`.")

        return super()._check_epsilon_delta(epsilon, delta)

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        return 0.0

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        raise NotImplementedError

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        if self._scale is None or self._noise_bound is None:
            self._scale = self.sensitivity / self.epsilon
            self._noise_bound = 0 if self._scale == 0 else \
                self._scale * np.log(1 + (np.exp(self.epsilon) - 1) / 2 / self.delta)

        if np.isnan(value):
            return float("nan")

        samples = 1

        while True:
            try:
                unif = self._rng.random(4 * samples)
            except TypeError:  # rng is secrets.SystemRandom
                unif = [self._rng.random() for _ in range(4 * samples)]
            noisy = self._scale * self._laplace_sampler(*np.array(unif).reshape(4, -1))

            if ((noisy >= - self._noise_bound) & (noisy <= self._noise_bound)).any():
                idx = np.argmax((noisy >= - self._noise_bound) & (noisy <= self._noise_bound))
                return value + noisy[idx]
            samples = min(100000, samples * 2)










# Previous code with full debugging print statements..!!



# # MIT License
# #
# # Copyright (C) IBM Corporation 2019
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# # documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# # rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# # persons to whom the Software is furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# # Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# # WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# # TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
# """
# The classic Laplace mechanism in differential privacy, and its derivatives.
# """
# from numbers import Real

# import numpy as np

# #from .laplace import nonnegativity
# from diffprivlib.mechanisms.base import DPMechanism, TruncationAndFoldingMixin
# from diffprivlib.utils import copy_docstring


# class Laplace(DPMechanism):
#     r"""
#     The classical Laplace mechanism in differential privacy.

#     First proposed by Dwork, McSherry, Nissim and Smith [DMNS16]_, with support for (relaxed)
#     :math:`(\epsilon,\delta)`-differential privacy [HLM15]_.

#     Samples from the Laplace distribution are generated using 4 uniform variates, as detailed in [HB21]_, to prevent
#     against reconstruction attacks due to limited floating point precision.

#     Parameters
#     ----------
#     epsilon : float
#         Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

#     delta : float, default: 0.0
#         Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
#         ``epsilon``.

#     sensitivity : float
#         The sensitivity of the mechanism.  Must be in [0, ∞).

#     random_state : int or RandomState, optional
#         Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
#         ``random_state`` has to be fixed to an integer.

#     References
#     ----------
#     .. [DMNS16] Dwork, Cynthia, Frank McSherry, Kobbi Nissim, and Adam Smith. "Calibrating noise to sensitivity in
#         private data analysis." Journal of Privacy and Confidentiality 7, no. 3 (2016): 17-51.

#     .. [HLM15] Holohan, Naoise, Douglas J. Leith, and Oliver Mason. "Differential privacy in metric spaces: Numerical,
#         categorical and functional data under the one roof." Information Sciences 305 (2015): 256-268.

#     .. [HB21] Holohan, Naoise, and Stefano Braghin. "Secure Random Sampling in Differential Privacy." arXiv preprint
#         arXiv:2107.10138 (2021).

#     """
#     def __init__(self, *, epsilon, delta=0.0, sensitivity, random_state=None):
#         super().__init__(epsilon=epsilon, delta=delta, random_state=random_state)
#         self.sensitivity = self._check_sensitivity(sensitivity)
#         self._scale = None

#     @classmethod
#     def _check_sensitivity(cls, sensitivity):
#         if not isinstance(sensitivity, Real):
#             raise TypeError("Sensitivity must be numeric")

#         if sensitivity < 0:
#             raise ValueError("Sensitivity must be non-negative")

#         return float(sensitivity)

#     def _check_all(self, value):
#         super()._check_all(value)
#         self._check_sensitivity(self.sensitivity)

#         if not isinstance(value, Real):
#             raise TypeError("Value to be randomised must be a number")

#         return True

#     def bias(self, value):
#         """Returns the bias of the mechanism at a given `value`.

#         Parameters
#         ----------
#         value : int or float
#             The value at which the bias of the mechanism is sought.

#         Returns
#         -------
#         bias : float or None
#             The bias of the mechanism at `value`.

#         """
#         return 0.0

#     def variance(self, value):
#         """Returns the variance of the mechanism at a given `value`.

#         Parameters
#         ----------
#         value : float
#             The value at which the variance of the mechanism is sought.

#         Returns
#         -------
#         bias : float
#             The variance of the mechanism at `value`.

#         """
#         self._check_all(0)

#         return 2 * (self.sensitivity / (self.epsilon - np.log(1 - self.delta))) ** 2

#     @staticmethod
#     def _laplace_sampler(unif1, unif2, unif3, unif4):
#         return np.log(1 - unif1) * np.cos(np.pi * unif2) + np.log(1 - unif3) * np.cos(np.pi * unif4)

#     # def randomise(self, value):
#     #     """Randomise `value` with the mechanism.

#     #     Parameters
#     #     ----------
#     #     value : float
#     #         The value to be randomised.

#     #     Returns
#     #     -------
#     #     float
#     #         The randomised value.

#     #     """
#     #     self._check_all(value)
#     #     print('[super.randomize] Inside the laplace class randomize, and Value: {}'.format(value))

#     #     scale = self.sensitivity / (self.epsilon - np.log(1 - self.delta))
#     #     print('[super.randomize] Inside the laplace class randomize, and Scale: {}'.format(scale))
#     #     standard_laplace = self._laplace_sampler(self._rng.random(), self._rng.random(), self._rng.random(),
#     #                                              self._rng.random())
#     #     print('[super.randomize] Inside the laplace class randomize, and Standard Laplace: {}'.format(standard_laplace))
#     #     print('[super.randomize] returns the new randomized value as: {}'.format(value - scale * standard_laplace))
#     #     noisy_value = value - scale * standard_laplace
#     #     return value - scale * standard_laplace

#     def nonnegativity(self, array, rho=0.0001, T=30):
#         """Enforces non-negativity on an array of integers, gradually reducing negative values.

#         Args:
#             array: The input array of integers.
#             rho: The threshold for absolute sum of negative numbers (default: 0.0001).
#             T: The maximum number of rounds to perform (default: 30).

#         Returns:
#             The modified array with non-negative values.
#         """

#         round_count = 0
#         print("\n\n----START---- [non-negativity] Initial array before non-negativity enforcement is:\n", array,"\n")
#         while round_count < T and abs(sum(x for x in array if x < 0)) > rho:
#             print(f"\n\n==============================[non-negativity] Inside the non-negativity enforcement loop. Round count is {round_count} :====================================")
#             round_count += 1
#             for i in range(len(array)):
#                 if array[i] < 0:
#                     print(f"\n[non-negativity] --- negative value found at position {i}:", array[i])
#                 else:
#                     print(f"\n[non-negativity] +++ positive value found at position {i}:", array[i])
#             print("\n\n")
#             negative_sum = sum(abs(x) for x in array if x < 0)
#             print("\n[non-negativity] Absolute Sum of negative numbers above is:", negative_sum)
#             positive_count = sum(1 for x in array if x > 0)  # Count positive values using generator expression
#             print("\n[non-negativity] Count of positive numbers above is:", positive_count)
            
#             try:
#                 height = negative_sum / positive_count  # Handle potential division by zero
#             except ZeroDivisionError:
#                 height = 0

#             print(f"\n[non-negativity] Height for round no.{round_count} is:", height)
#             print("\n\n")
#             for i in range(len(array)):
#                 if array[i] > 0:
#                     print(f"\n[non-negativity] +++ positive value at position {i}:", array[i])
#                     array[i] -= height
#                     print(f"\n[non-negativity] +++ new value at position {i} after subtraction of height :", array[i])
#                 elif array[i] < 0:
#                     print(f"\n[non-negativity] --- negative value at position {i}:", array[i])
#                     array[i] = 0
#                     print(f"\n[non-negativity] --- new value at position {i} after setting to 0 :", array[i])
        
        
#         print("\n\n[non-negativity] One of the 3 main conditions(mostly, No negative numbers found..!!) is not satisfied, so we are setting any of the remaining negative values to 0.\n")

#         # Set any remaining negative values to 0
#         for i in range(len(array)):
#             if array[i] < 0:
#                 print(f"\n[non-negativity] --- previous negative value at position {i}:", array[i])
#                 array[i] = 0
#                 print(f"\n[non-negativity] --- new value at position {i} after setting to 0 :", array[i])

#         print("\n----END---- [non-negativity] Final array after non-negativity enforcement is:\n", array,"\n")
#         return array


#     def randomise(self, values):
#         """Randomise `values` with the mechanism.

#         Parameters
#         ----------
#         values : list or array-like
#             The values to be randomised.

#         Returns
#         -------
#         array-like
#             The array of randomised values.

#         """
#         result_values = []
#         for value in values:
#             self._check_all(value)
        
#         print('[super.randomize] Inside the laplace class randomize, and Values: {}'.format(values))
        
#         for value in values:
#             scale = self.sensitivity / (self.epsilon - np.log(1 - self.delta))
#             print('[super.randomize] Inside the laplace class randomize, and Scale: {}'.format(scale))
#             standard_laplace = self._laplace_sampler(self._rng.random(), self._rng.random(), self._rng.random(),
#                                                     self._rng.random())
#             print('[super.randomize] Inside the laplace class randomize, and Standard Laplace: {}'.format(standard_laplace))
#             noisy_value = value - scale * standard_laplace
#             print('[super.randomize] returns the new randomized value as: {}'.format(noisy_value))
#             result_values.append(noisy_value)
            
#         print('\n[super.randomize] returns the new randomized values as: {}'.format(result_values))
#         #return result_values
#         print("\n[super.randomize] The noisy values are sent for non-negativity enforcing..!")
#         noisy_values = self.nonnegativity(result_values)
#         return noisy_values


    
    
    
#     # def randomise(self, values, threshold=0.3, T=30):
#     #     """Randomise a list of values with the mechanism while enforcing non-negativity.

#     #     Parameters
#     #     ----------
#     #     values : list or array-like
#     #         The values to be randomised.
#     #     threshold : float, optional
#     #         The threshold for stopping the non-negativity loop. Default is 0.3.
#     #     T : int, optional
#     #         The maximum number of rounds for non-negativity. Default is 30.

#     #     Returns
#     #     -------
#     #     array-like
#     #         The array of randomised values.

#     #     """
#     #     # Check all values in the list
#     #     for value in values:
#     #         self._check_all(value)

#     #     print('[super.randomize] Inside the laplace class randomize, and Values: {}'.format(values))

#     #     scale = self.sensitivity / (self.epsilon - np.log(1 - self.delta))
#     #     print('[super.randomize] Inside the laplace class randomize, and Scale: {}'.format(scale))

#     #     sum_negative = 0  # Variable to store the sum of negative numbers
#     #     num_positive = 0  # Variable to count the number of positive numbers
#     #     T_counter = 0  # Variable to count the number of rounds of non-negativity

#     #     while True:
#     #         randomized_values = []  # List to store the randomised values

#     #         for value in values:
#     #             standard_laplace = self._laplace_sampler(self._rng.random(), self._rng.random(), self._rng.random(),
#     #                                                     self._rng.random())
#     #             randomized_value = value - scale * standard_laplace

#     #             if randomized_value < 0:
#     #                 sum_negative += abs(randomized_value)
#     #                 num_positive += 1
#     #                 randomized_value = 0  # Set negative value to 0 for now

#     #             randomized_values.append(randomized_value)

#     #         if sum_negative / num_positive > threshold or T_counter >= T:
#     #             return np.maximum(0, randomized_values)  # If threshold is exceeded or T rounds reached, return values with negative values set to 0
#     #         else:
#     #             print("[super.randomize] SUM_negative: {}, NUM_positive: {}".format(sum_negative, num_positive))
#     #             values = np.maximum(0, randomized_values - sum_negative / num_positive)  # Adjust values for non-negativity
#     #             sum_negative = 0  # Reset sum_negative for the next round
#     #             num_positive = 0  # Reset num_positive for the next round
#     #             T_counter += 1
#     #         return randomized_values



# class LaplaceTruncated(Laplace, TruncationAndFoldingMixin):
#     r"""
#     The truncated Laplace mechanism, where values outside a pre-described domain are mapped to the closest point
#     within the domain.

#     Parameters
#     ----------
#     epsilon : float
#         Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

#     delta : float, default: 0.0
#         Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
#         ``epsilon``.

#     sensitivity : float
#         The sensitivity of the mechanism.  Must be in [0, ∞).

#     lower : float
#         The lower bound of the mechanism.

#     upper : float
#         The upper bound of the mechanism.

#     random_state : int or RandomState, optional
#         Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
#         ``random_state`` has to be fixed to an integer.

#     """
#     def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper, random_state=None):
#         super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
#         TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

#     @copy_docstring(Laplace.bias)
#     def bias(self, value):
#         self._check_all(value)

#         shape = self.sensitivity / self.epsilon
#         print('@@@@ Inside the laplace truncated class, and Shape: {}'.format(shape))
#         print('@@@@ Inside the laplace truncated class, and Value: {}'.format(value))
#         print('@@@@ Inside the laplace truncated class, and Lower: {}'.format(self.lower))
#         print('@@@@ Inside the laplace truncated class, and Upper: {}'.format(self.upper))
#         print('@@@@ Inside the laplace truncated class, and returns ----: {}',shape / 2 * (np.exp((self.lower - value) / shape) - np.exp((value - self.upper) / shape)))
#         return shape / 2 * (np.exp((self.lower - value) / shape) - np.exp((value - self.upper) / shape))

#     @copy_docstring(Laplace.variance)
#     def variance(self, value):
#         self._check_all(value)

#         shape = self.sensitivity / self.epsilon

#         variance = value ** 2 + shape * (self.lower * np.exp((self.lower - value) / shape)
#                                          - self.upper * np.exp((value - self.upper) / shape))
#         variance += (shape ** 2) * (2 - np.exp((self.lower - value) / shape)
#                                     - np.exp((value - self.upper) / shape))

#         variance -= (self.bias(value) + value) ** 2

#         return variance

#     def _check_all(self, value):
#         Laplace._check_all(self, value)
#         TruncationAndFoldingMixin._check_all(self, value)

#         return True

#     @copy_docstring(Laplace.randomise)
#     # def randomise(self, value):
#     #     print('----START(randomize)---- Inside the laplace truncated randomize class, and original value is: {}'.format(value))
#     #     self._check_all(value)
#     #     #print('@@@@ Inside the laplace truncated randomize class, and Value: {}'.format(value))
#     #     noisy_value = super().randomise(value)
#     #     print('----END(randomize)---- The noisy value returned after randomizing the original value is: {}'.format(noisy_value))
#     #     print('@@@@ Inside the laplace truncated randomize class, and returns-------------->>>>>>>: ',self._truncate(noisy_value))
#     #     return self._truncate(noisy_value)
    
    
#     def randomise(self, values):
#         """Randomise a list of values with the mechanism.

#         Parameters
#         ----------
#         values : list or array-like
#             The values to be randomised.

#         Returns
#         -------
#         array-like
#             The array of randomised values.

#         """
#         print('----START(randomize)---- Inside the laplace truncated randomize class, and original values are: {}'.format(values))
        
#         # Ensure all values in the list pass the check
#         for value in values:
#             self._check_all(value)
        
#         # Send the entire list of values for randomization to the superclass
#         noisy_values = super().randomise(values)
        
#         print('----END(randomize)---- The noisy values returned after randomizing the original values are: {}'.format(noisy_values))
#         #print('@@@@ Inside the laplace truncated randomize class, and returns-------------->>>>>>>: ', self._truncate(noisy_values))
        
#         return noisy_values
#         #return self._truncate(noisy_values)



# class LaplaceFolded(Laplace, TruncationAndFoldingMixin):
#     r"""
#     The folded Laplace mechanism, where values outside a pre-described domain are folded around the domain until they
#     fall within.

#     Parameters
#     ----------
#     epsilon : float
#         Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

#     delta : float, default: 0.0
#         Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
#         ``epsilon``.

#     sensitivity : float
#         The sensitivity of the mechanism.  Must be in [0, ∞).

#     lower : float
#         The lower bound of the mechanism.

#     upper : float
#         The upper bound of the mechanism.

#     random_state : int or RandomState, optional
#         Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
#         ``random_state`` has to be fixed to an integer.

#     """
#     def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper, random_state=None):
#         super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
#         TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

#     @copy_docstring(Laplace.bias)
#     def bias(self, value):
#         self._check_all(value)

#         shape = self.sensitivity / self.epsilon

#         bias = shape * (np.exp((self.lower + self.upper - 2 * value) / shape) - 1)
#         bias /= np.exp((self.lower - value) / shape) + np.exp((self.upper - value) / shape)

#         return bias

#     @copy_docstring(DPMechanism.variance)
#     def variance(self, value):
#         raise NotImplementedError

#     def _check_all(self, value):
#         super()._check_all(value)
#         TruncationAndFoldingMixin._check_all(self, value)

#         return True

#     @copy_docstring(Laplace.randomise)
#     def randomise(self, value):
#         self._check_all(value)

#         noisy_value = super().randomise(value)
#         return self._fold(noisy_value)



# class LaplaceBoundedDomain(LaplaceTruncated):
#     r"""
#     The bounded Laplace mechanism on a bounded domain.  The mechanism draws values directly from the domain using
#     rejection sampling, without any post-processing [HABM20]_.

#     Parameters
#     ----------
#     epsilon : float
#         Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

#     delta : float, default: 0.0
#         Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
#         ``epsilon``.

#     sensitivity : float
#         The sensitivity of the mechanism.  Must be in [0, ∞).

#     lower : float
#         The lower bound of the mechanism.

#     upper : float
#         The upper bound of the mechanism.

#     random_state : int or RandomState, optional
#         Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
#         ``random_state`` has to be fixed to an integer.

#     References
#     ----------
#     .. [HABM20] Holohan, Naoise, Spiros Antonatos, Stefano Braghin, and Pól Mac Aonghusa. "The Bounded Laplace Mechanism
#         in Differential Privacy." Journal of Privacy and Confidentiality 10, no. 1 (2020).

#     """
#     def _find_scale(self):
#         eps = self.epsilon
#         delta = self.delta
#         diam = self.upper - self.lower
#         delta_q = self.sensitivity

#         def _delta_c(shape):
#             if shape == 0:
#                 return 2.0
#             return (2 - np.exp(- delta_q / shape) - np.exp(- (diam - delta_q) / shape)) / (1 - np.exp(- diam / shape))

#         def _f(shape):
#             return delta_q / (eps - np.log(_delta_c(shape)) - np.log(1 - delta))

#         left = delta_q / (eps - np.log(1 - delta))
#         right = _f(left)
#         old_interval_size = (right - left) * 2

#         while old_interval_size > right - left:
#             old_interval_size = right - left
#             middle = (right + left) / 2

#             if _f(middle) >= middle:
#                 left = middle
#             if _f(middle) <= middle:
#                 right = middle

#         return (right + left) / 2

#     def effective_epsilon(self):
#         r"""Gets the effective epsilon of the mechanism, only for strict :math:`\epsilon`-differential privacy.  Returns
#         ``None`` if :math:`\delta` is non-zero.

#         Returns
#         -------
#         float
#             The effective :math:`\epsilon` parameter of the mechanism.  Returns ``None`` if `delta` is non-zero.

#         """
#         if self._scale is None:
#             self._scale = self._find_scale()

#         if self.delta > 0.0:
#             return None

#         return self.sensitivity / self._scale

#     @copy_docstring(Laplace.bias)
#     def bias(self, value):
#         self._check_all(value)

#         if self._scale is None:
#             self._scale = self._find_scale()

#         bias = (self._scale - self.lower + value) / 2 * np.exp((self.lower - value) / self._scale) \
#             - (self._scale + self.upper - value) / 2 * np.exp((value - self.upper) / self._scale)
#         bias /= 1 - np.exp((self.lower - value) / self._scale) / 2 \
#             - np.exp((value - self.upper) / self._scale) / 2

#         return bias

#     @copy_docstring(Laplace.variance)
#     def variance(self, value):
#         self._check_all(value)

#         if self._scale is None:
#             self._scale = self._find_scale()

#         variance = value**2
#         variance -= (np.exp((self.lower - value) / self._scale) * (self.lower ** 2)
#                      + np.exp((value - self.upper) / self._scale) * (self.upper ** 2)) / 2
#         variance += self._scale * (self.lower * np.exp((self.lower - value) / self._scale)
#                                    - self.upper * np.exp((value - self.upper) / self._scale))
#         variance += (self._scale ** 2) * (2 - np.exp((self.lower - value) / self._scale)
#                                           - np.exp((value - self.upper) / self._scale))
#         variance /= 1 - (np.exp(-(value - self.lower) / self._scale)
#                          + np.exp(-(self.upper - value) / self._scale)) / 2

#         variance -= (self.bias(value) + value) ** 2

#         return variance

#     @copy_docstring(Laplace.randomise)
#     def randomise(self, value):
#         self._check_all(value)

#         if self._scale is None:
#             self._scale = self._find_scale()

#         value = max(min(value, self.upper), self.lower)
#         if np.isnan(value):
#             return float("nan")

#         samples = 1

#         while True:
#             try:
#                 unif = self._rng.random(4 * samples)
#             except TypeError:  # rng is secrets.SystemRandom
#                 unif = [self._rng.random() for _ in range(4 * samples)]
#             noisy = value + self._scale * self._laplace_sampler(*np.array(unif).reshape(4, -1))

#             if ((noisy >= self.lower) & (noisy <= self.upper)).any():
#                 idx = np.argmax((noisy >= self.lower) & (noisy <= self.upper))
#                 return noisy[idx]
#             samples = min(100000, samples * 2)


# class LaplaceBoundedNoise(Laplace):
#     r"""
#     The Laplace mechanism with bounded noise, only applicable for approximate differential privacy (delta > 0)
#     [GDGK18]_.

#     Epsilon must be strictly positive, `epsilon` > 0. `delta` must be strictly in the interval (0, 0.5).
#      - For zero `epsilon`, use :class:`.Uniform`.
#      - For zero `delta`, use :class:`.Laplace`.

#     Parameters
#     ----------
#     epsilon : float
#         Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

#     delta : float
#         Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 0.5).

#     sensitivity : float
#         The sensitivity of the mechanism.  Must be in [0, ∞).

#     random_state : int or RandomState, optional
#         Controls the randomness of the mechanism.  To obtain a deterministic behaviour during randomisation,
#         ``random_state`` has to be fixed to an integer.

#     References
#     ----------
#     .. [GDGK18] Geng, Quan, Wei Ding, Ruiqi Guo, and Sanjiv Kumar. "Truncated Laplacian Mechanism for Approximate
#         Differential Privacy." arXiv preprint arXiv:1810.00877v1 (2018).

#     """
#     def __init__(self, *, epsilon, delta, sensitivity, random_state=None):
#         super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
#         self._noise_bound = None

#     @classmethod
#     def _check_epsilon_delta(cls, epsilon, delta):
#         if epsilon == 0:
#             raise ValueError("Epsilon must be strictly positive. For zero epsilon, use :class:`.Uniform`.")

#         if isinstance(delta, Real) and not 0 < delta < 0.5:
#             raise ValueError("Delta must be strictly in the interval (0,0.5). For zero delta, use :class:`.Laplace`.")

#         return super()._check_epsilon_delta(epsilon, delta)

#     @copy_docstring(Laplace.bias)
#     def bias(self, value):
#         return 0.0

#     @copy_docstring(DPMechanism.variance)
#     def variance(self, value):
#         raise NotImplementedError

#     @copy_docstring(Laplace.randomise)
#     def randomise(self, value):
#         self._check_all(value)

#         if self._scale is None or self._noise_bound is None:
#             self._scale = self.sensitivity / self.epsilon
#             self._noise_bound = 0 if self._scale == 0 else \
#                 self._scale * np.log(1 + (np.exp(self.epsilon) - 1) / 2 / self.delta)

#         if np.isnan(value):
#             return float("nan")

#         samples = 1

#         while True:
#             try:
#                 unif = self._rng.random(4 * samples)
#             except TypeError:  # rng is secrets.SystemRandom
#                 unif = [self._rng.random() for _ in range(4 * samples)]
#             noisy = self._scale * self._laplace_sampler(*np.array(unif).reshape(4, -1))

#             if ((noisy >= - self._noise_bound) & (noisy <= self._noise_bound)).any():
#                 idx = np.argmax((noisy >= - self._noise_bound) & (noisy <= self._noise_bound))
#                 return value + noisy[idx]
#             samples = min(100000, samples * 2)
