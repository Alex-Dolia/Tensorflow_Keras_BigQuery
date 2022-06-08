The task is given a tensor of strings with logical expression convert infix of of expression in postfix form.
the postfix foem is easier for caclulating logical expressions.

For example, we have the following tensor:

[[b'( 1 OR 2 )  AND 3 OR NOT ( 4 OR 5 )']
 [b'( 10 OR 20 ) AND 30 OR NOT ( 40 OR 50 )']
 [b'( 100 OR 200 ) AND 300 OR NOT ( 400 OR 500 )']]

After running the following script:

input_layer  = Input(shape = (1), dtype= tf.string)
#
output_layer = Precision_AD()(input_layer)  
#
model = Model(inputs = input_layer, outputs = output_layer)
#
pred = model.predict(inputs, batch_size = 150)
print(pred)

we get the postfix form of the above logical expression:

[[b' 1 2 OR 3 AND 4 5 OR NOT OR']
 [b' 10 20 OR 30 AND 40 50 OR NOT OR']
 [b' 100 200 OR 300 AND 400 500 OR NOT OR']]