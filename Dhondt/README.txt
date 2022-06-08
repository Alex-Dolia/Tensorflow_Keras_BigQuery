Both DHondt_TensorFlow_KERAS_1D and DHondt_TensorFlow_KERAS_2D compute Dhondt algorithm
similar to the way it is described in https://gist.github.com/brunosan/96288a8612894fca718aacbcc501ee09

The difference between DHondt_TensorFlow_KERAS_1D and DHondt_TensorFlow_KERAS_2D
that in the first case the input and the output is 1D column vector but for the second
case it is 2D matrix.

For example, DHondt_TensorFlow_KERAS_1D inputs is the following: 
[[0.29999998]
 [0.7       ]
 [0.39999998]
 [0.59999996]
 [5.        ]
 [0.29999998]
 [0.7       ]
 [0.39999998]
 [0.59999996]
 [0.5       ]
 [0.5       ]]
and DHondt_TensorFlow_KERAS_1D output:
[[0.]
 [1.]
 [0.]
 [0.]
 [8.]
 [0.]
 [1.]
 [0.]
 [0.]
 [0.]
 [0.]]

But for DHondt_TensorFlow_KERAS_2D inputs containing the same values as above is 2D matrix:
[[0.29999998 0.7        0.39999998 0.59999996]
 [5.         0.29999998 0.7        0.39999998]
 [0.59999996 0.5        0.5        0.        ]]
and DHondt_TensorFlow_KERAS_2D output is the following:
[[0. 1. 0. 0.]
 [8. 0. 1. 0.]
 [0. 0. 0. 0.]]

You can see if you reshape the output of DHondt_TensorFlow_KERAS_2D to 1D column vector
then you will get exactly the same output as DHondt_TensorFlow_KERAS_1D