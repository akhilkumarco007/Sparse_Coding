import tensorflow as tf
import numpy as np


def variable_creator(name, shape):
    with tf.name_scope(name=name):
        intial = tf.truncated_normal_initializer(stddev=0.01)
        return tf.get_variable(name=name, shape=shape, initializer=intial)


def loss(y, x, d, lamb, name):
    with tf.name_scope(name=name):
        diff_tensor = tf.subtract(y, tf.matmul(d, x))
        first_term = tf.norm(diff_tensor, name='First_Term')
        second_term = lamb * tf.norm(x, ord=1, name='Second_Term')
        return first_term + second_term


def normalize_input(data, name):
    with tf.name_scope(name=name):
        normalized = np.transpose(data).astype(float)
        max_min = normalized[np.where(normalized != 0)]
        for i in range(len(normalized)):
            normalized[i] = (normalized[i] - np.min(max_min)) / (np.max(max_min) - np.min(max_min))
        return np.transpose(normalized), np.max(max_min), np.min(max_min)

# def normalize_input(data, name):
#     with tf.name_scope(name=name):
#         normalized = np.transpose(data).astype(float)
#         for i in range(len(normalized)):
#             normalized[i] = (normalized[i] - np.min(normalized[i])) / (np.max(normalized[i]) - np.min(normalized[i]))
#         return np.transpose(normalized)

def de_normalized(output, max, min):
    return (output * (max - min)) + min
