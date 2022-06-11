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
then you will get exactly the same output as DHondt_TensorFlow_KERAS_1D.

But DHondt_TensorFlow_KERAS_2D is better suit for using in BigQuery - it can take the row as input and 
return the row of the same size as output. We do not recommnend using DHondt_TensorFlow_KERAS_1D for BigQuery
but it can be used in python script.

In order to perform ML.PREDICT Big Query splits table in batch that have the following constrain:
1) the total number of element in the batch is less or equal to 234000;
2) the number of rows (#row) in a batch should be less or equal to 512 subject to #row * #cols <= 234000 
   where #cols is the number of column in the batch;
3) the number of columns or elements of array in the row is less or equal to 2743.
   Note that, the larger values might increase the size of Tensorflow model that are not allowed in BigQuery.
   Models are limited to 250MB in size but if you increase the number of elements in the row for the model you increase its size
   even it does not actually require more memory maybe it reserve the space.
   See the following link for details: https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create-tensorflow#:~:text=Models%20are%20limited%20to%20250MB,of%20TensorFlow%20are%20not%20supported.
