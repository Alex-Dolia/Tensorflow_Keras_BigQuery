-- this script will load Tensorflow model into BigQuery
-- https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create-tensorflow#limitations
--
CREATE OR REPLACE MODEL `your-project-id.your-dataset-id.DHondt_TensorFlow_KERAS_2D`
          OPTIONS ( MODEL_TYPE='TENSORFLOW',
                    MODEL_PATH='gs://your-bucket/DHondt_TensorFlow_KERAS_2D/*'
                  )
-- instead of `your-project-id.your-dataset-id` and `your-bucket` you need put your own strings.
-- after model loaded check its schema - it will give you the name of inputs and output that 
-- you need to use when you make prediction using ML.PREDICT