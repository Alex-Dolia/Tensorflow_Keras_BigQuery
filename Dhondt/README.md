# Deploying DHondt_TensorFlow_KERAS_1D and DHondt_TensorFlow_KERAS_2D to BigQuery

This repository includes TensorFlow Keras implementations of the D'Hondt algorithm that can be deployed to BigQuery ML using `MODEL_TYPE='TENSORFLOW'`. The 2D version (`DHondt_TensorFlow_KERAS_2D`) is especially suited for this use case due to its row-wise input/output format.

## Deploying to BigQuery

Once the model is trained and saved locally, upload it to a GCS bucket and use the following SQL to deploy:

```sql
-- This script will load TensorFlow model into BigQuery
-- Reference: https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create-tensorflow#limitations

CREATE OR REPLACE MODEL `your-project-id.your-dataset-id.DHondt_TensorFlow_KERAS_2D`
OPTIONS (
  MODEL_TYPE = 'TENSORFLOW',
  MODEL_PATH = 'gs://your-bucket/DHondt_TensorFlow_KERAS_2D/*'
);
```

> Replace `your-project-id.your-dataset-id` and `your-bucket` with your actual project, dataset, and Cloud Storage bucket.

### After Deployment

You can verify the input/output schema using:

```sql
SELECT * FROM ML.MODEL_SCHEMA(`your-project-id.your-dataset-id.DHondt_TensorFlow_KERAS_2D`);
```

### Making Predictions in BigQuery

Once deployed, you can use the model with `ML.PREDICT` like this:

```sql
SELECT *
FROM ML.PREDICT(
  MODEL `your-project-id.your-dataset-id.DHondt_TensorFlow_KERAS_2D`,
  TABLE `your-project-id.your-dataset-id.your_input_table`
);
```

---

# DHondt TensorFlow Keras Implementations (1D and 2D)

This repository provides two TensorFlow Keras implementations of the D'Hondt algorithm, inspired by [this gist](https://gist.github.com/brunosan/96288a8612894fca718aacbcc501ee09).

## Overview

- `DHondt_TensorFlow_KERAS_1D`: Accepts a 1D column vector as input and returns a 1D column vector as output.
- `DHondt_TensorFlow_KERAS_2D`: Accepts a 2D matrix as input (row-wise vote data) and returns a 2D matrix of the same shape with allocated seats per input row.

These implementations are suitable for use in Python scripts and BigQuery ML, respectively.

---

## Input and Output Examples

### 1D Version (`DHondt_TensorFlow_KERAS_1D`)
**Input:**
```plaintext
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
```

**Output:**
```plaintext
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
```

---

### 2D Version (`DHondt_TensorFlow_KERAS_2D`)
**Input (same values reshaped):**
```plaintext
[[0.29999998 0.7        0.39999998 0.59999996]
 [5.         0.29999998 0.7        0.39999998]
 [0.59999996 0.5        0.5        0.        ]]
```

**Output:**
```plaintext
[[0. 1. 0. 0.]
 [8. 0. 1. 0.]
 [0. 0. 0. 0.]]
```

**Note:** Reshaping the 2D output back into a column vector reproduces the exact 1D output.

---

## Use in BigQuery ML

`DHondt_TensorFlow_KERAS_2D` is the recommended model for BigQuery ML due to its row-wise processing and compatibility with BigQuery's batch constraints:

### BigQuery Constraints for `ML.PREDICT`
1. **Total batch elements ≤ 234,000**
2. **Max rows per batch ≤ 512**, subject to:  
   `#rows × #columns ≤ 234,000`
3. **Max columns (or array elements per row) ≤ 2,743**

> Larger values may increase the size of the TensorFlow model, which must remain ≤ 250MB. Even if memory is not fully utilized, TensorFlow may reserve space.

For more details, refer to the [BigQuery ML TensorFlow model documentation](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create-tensorflow#:~:text=Models%20are%20limited%20to%20250MB,of%20TensorFlow%20are%20not%20supported.).

---

## Example Code

The following demonstrates how to use the 2D model:

```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate

# Custom Layer
class DHondt_TensorFlow_KERAS_2D(tf.keras.layers.Layer):                        
    def __init__(self, model_column_number, **kwargs):
        super().__init__(**kwargs)
        self.model_column_number = model_column_number

    def get_config(self):
        config = super().get_config()
        config['model_column_number'] = self.model_column_number
        return config

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return [(None, 1) for _ in range(self.model_column_number)]

    @tf.function
    def call(self, inputs):
        inputs = tf.reshape(inputs, [-1])
        inputs = tf.squeeze(inputs)
        nSeats = tf.math.reduce_sum(inputs)
        seats = tf.cast(tf.math.floor(inputs), tf.float32)

        if tf.greater(nSeats, 0):
            votes = tf.divide(inputs, nSeats)
            nSeats = tf.cast(nSeats, tf.int32)
            allocated = tf.cast(tf.reduce_sum(seats), tf.int32)
            t_votes = tf.identity(votes)

            while tf.less(allocated, nSeats):
                i = tf.math.argmax(t_votes)
                update = tf.gather(seats, i)
                update = tf.add(update, 1.0)
                seats = tf.tensor_scatter_nd_update(seats, [[i]], [update])
                allocated = tf.add(allocated, 1)
                t_votes = tf.tensor_scatter_nd_update(
                    t_votes, [[i]],
                    [tf.gather(votes, i) / (update + 1.0)]
                )

            seats = tf.reshape(seats, [-1, self.model_column_number])
        else:
            seats = tf.reshape(tf.zeros_like(inputs), [-1, self.model_column_number])

        return [seats[:, i:(i + 1)] for i in range(self.model_column_number)]

# Example usage
if __name__ == "__main__":
    model_column_number = 4
    input_layer = Input(shape=(model_column_number,))
    output_layer = DHondt_TensorFlow_KERAS_2D(model_column_number)(input_layer)
    output_layer = Concatenate()(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    total_seats = 10
    inputs = tf.constant([
        [0.03, 0.07, 0.04, 0.06],
        [0.5, 0.03, 0.07, 0.04],
        [0.06, 0.05, 0.05, 0.0]
    ]) * total_seats

    print("Inputs:")
    print(inputs)

    seats = model.predict(inputs)
    print("Allocated Seats:")
    print(seats)
    print("Reshaped Output:")
    print(tf.reshape(seats, [-1, 1]))
```

---

## Recommendation

- Use **`DHondt_TensorFlow_KERAS_2D`** for **BigQuery ML predictions**.
- Use **`DHondt_TensorFlow_KERAS_1D`** only for **local or script-based processing**.

---

## References

- [Original D'Hondt Gist](https://gist.github.com/brunosan/96288a8612894fca718aacbcc501ee09)
- [BigQuery ML TensorFlow Model Constraints](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create-tensorflow#:~:text=Models%20are%20limited%20to%20250MB,of%20TensorFlow%20are%20not%20supported.)
