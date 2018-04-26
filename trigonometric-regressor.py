# Uses a DNN regressor to model a trigonometric equation.
#
# This serves as a minimalistic reference implementation that can be built on for other problems.
# It performs an examination of different DNN structures, up to 4 hidden layers.
#
# Use tensorboard by using the following command in a shell:
#   tensorboard --logdir="model_dir"
#
# Initial implemention by Dag Erlandsen / Spinning Owl AS april 2018.
#
# MIT license

import tensorflow as tf
import numpy as np
import math
import time

class ConfigPermutation:
    def __init__(self):
        SIZES = [0, 8, 256]
        self.permutations = []

        for s4 in SIZES:
            for s3 in SIZES:
                for s2 in SIZES:
                    for s1 in SIZES:
                        config = [n for n in [s1, s2, s3, s4] if n != 0]
                        if len(config)>0 and not config in self.permutations:
                            self.permutations.append(config)

    def is_empty(self):
        return len(self.permutations) == 0

    def get_next_config(self):
        return self.permutations.pop()

def create_dataset(size):
    inputs = np.linspace(0, 1, size)
    np.random.shuffle(inputs)
    output = np.array([math.sin(a*PI*2)+math.cos((a+0.2)*PI*4) for a in inputs])    # Trigonometric equation
    return inputs, output

PI = 3.14159265359
TRAINING_RATIO = 0.75   # This much of the dataset will be used for training
DATASET_SIZE = 100

inputs, output = create_dataset(DATASET_SIZE)
bound_index = int(TRAINING_RATIO*len(inputs))
inputs_training = inputs[:bound_index]
inputs_verification = inputs[bound_index:]
output_training = output[:bound_index]
output_verification = output[bound_index:]

column =  tf.feature_column.numeric_column('x')

configs = ConfigPermutation()
while not configs.is_empty():
    start_time = time.time()
    config = configs.get_next_config()

    regressor = tf.estimator.DNNRegressor(
        feature_columns=[column],
        hidden_units=config,
        model_dir="model_dir\\{}".format(config))

    train_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": inputs_training},
        y=output_training, 
        shuffle=True,
        num_epochs=2000,
        batch_size=len(inputs_training))
    train_spec = tf.estimator.TrainSpec(input_fn=train_input, max_steps=2000)

    eval_input = tf.estimator.inputs.numpy_input_fn(
        x={"x": inputs_verification},
        y=output_verification, 
        shuffle=True,
        batch_size=len(inputs_verification))
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input, steps=1)
    
    tf.estimator.train_and_evaluate(regressor, train_spec, eval_spec)

    predict_input = tf.estimator.inputs.numpy_input_fn(
         x={"x": inputs_verification},
         num_epochs=1, shuffle=False)
    results = list(regressor.predict(predict_input))

    average_error = 0   
    for i in range(len(results)):
         average_error += abs(results[i]["predictions"]-output_verification[i])     # Could use RMSE or similar, but this is more intuitive
    average_error /= len(results)

    delta_time = time.time() - start_time;
    print("{} in {:.3f} seconds. Average prediction error: {}".format(config, delta_time, average_error))