## Linear Regression output for ANN

### Best Model Output for Sigmoid:

_INPUT_: [1, 2, 3, 4, 5, 6]

_TARGET_: [2, 4, 6, 8, 10, 12]

_TRUE FUNCTION_: y = 2x

_HIDDEN LAYERS_: [7, 3, 5, 8]

_LEARNING RATE_: 0.001

_EPOCHS_: 10000000

_PREDICTED RESULTS_: [2.0001491407699263, 3.999407559224627, 6.001430215816066, 7.998283093136383, 10.000683406191442, 12.000196708027705]

_EXPECTED TARGET_: [2, 4, 6, 8, 10, 12]

_ERROR_: 5.872253571105137e-06

### Best Model Output for Leaky ReLU:

_INPUT_: [1, 2, 3, 4, 5, 6] 

_TARGET_: [2, 4, 6, 8, 10, 12]

_LAYERS_: [10, 6, 1, 7]

_LEARNING RATE_: 0.01 

_EPOCHS_: 1000

_Activation_: Leaky ReLU

_PREDICTED RESULTS_: [1.9999999999999991, 3.9999999999999982, 5.999999999999998, 7.9999999999999964, 9.999999999999996, 11.999999999999996]

_EXPECTED TARGET_: [2, 4, 6, 8, 10, 12]

_ERROR_: 4.4965071597597673e-29


## Notes
1. Deep models generally produce results closer to the target. However, model still unsuitable for predicting values outside of the provided dataset.
2. Input data needs to be well distributed with few outliers. Outliers drastically affect model performance and can destroy predictive ability.
3. Sometimes, Model will return same result for all inputs. This result is usually the mean of the dataset. Not sure why this happens but resolved
by using smaller learning rate and higher training epochs.
4. Activation Function really matters, affects the speed of training. Leaky ReLU returns an optimised linear regression function much faster and much more reliably compared to Sigmoid.

## Features to implement
1. Try out a binary classifier model
2. Implement ANN using other activation functions. [DONE]
3. Implement different activation function for each layer instead of one type of function for all layers.
4. Add new gradient descent functions such as momentum gradient descent, stochastic etc.
5. Implement feature to merge multiple pre-trained ANNs. Could be useful in feature engineering for complex ML problems
