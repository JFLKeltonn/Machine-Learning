## Linear Regression output for ANN

### Best Model Output:

_INPUT_: [1, 2, 3, 4, 5, 6]

_TARGET_: [2, 4, 6, 8, 10, 12]

_TRUE FUNCTION_: y = 2x

_HIDDEN LAYERS_: [7, 3, 5, 8]

_LEARNING RATE_: 0.001

_EPOCHS_: 10000000

_PREDICTED RESULTS_: [2.0001491407699263, 3.999407559224627, 6.001430215816066, 7.998283093136383, 10.000683406191442, 12.000196708027705]

_EXPECTED TARGET_: [2, 4, 6, 8, 10, 12]

_ERROR_: 5.872253571105137e-06

## Notes
1. Deep models generally produce results closer to the target. However, model still unsuitable for predicting values outside of the provided dataset.
2. Input data needs to be well distributed with few outliers. Outliers drastically affect model performance and can destroy predictive ability.
3. Sometimes, Model will return same result for all inputs. This result is usually the mean of the dataset. Not sure why this happens but resolved
by using smaller learning rate and higher training epochs.
4. There may be a trick to using the correct activation function for performance based on the type of problem. Need to experiment.

## Features to implement
1. Try out a binary classifier model
2. Implement ANN using other activation functions.
3. Implement different activation function for each layer instead of one type of function for all layers.
