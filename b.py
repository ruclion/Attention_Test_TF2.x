import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

keys = list(range(100))
length = 111
output_length = 222

def load_data(key):
    #print('Key %d loaded.' % key)
    data = np.expand_dims(np.sin(np.array(range(0, length)) * np.pi * np.cos(key) / 180 * 2), axis=-1)
    data1 = np.expand_dims(np.sin(np.array(range(0, output_length)) * length / output_length * np.pi * np.cos(key) / 180 * 2), axis=-1)
    return data, data1

sample = load_data(1)

def data_generator():
    for key in iter(keys):
        yield load_data(key)

dataset = tf.data.Dataset.from_generator(
        generator=data_generator,
        output_types=(tf.float64, tf.float64),
        output_shapes=(tf.TensorShape((length, 1)), tf.TensorShape((output_length, 1)))
        )

dataset = dataset.shuffle(8).batch(16)
#for i in dataset.take(2):
#    print(i)

class BahdanauAttention(Layer):

    def __init__(self, units):
        super().__init__()
        self.query_dense = tf.keras.layers.Dense(units)
        self.memory_dense = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.memory_dense(values) + self.query_dense(query)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class LocationSensitiveAttention(BahdanauAttention):

    class LocationLayer(Layer):

        def __init__(self, units, conv1d_filters, conv1d_kernel_size):
            super().__init__()
            self.conv1d = Conv1D(conv1d_filters, kernel_size=conv1d_kernel_size, padding='same', use_bias=False, bias_initializer=tf.zeros_initializer())
            self.dense = Dense(units, use_bias=False, activation='tanh')

        def call(self, inputs):
            x = self.conv1d(inputs)
            x = self.dense(x)
            return x

    def __init__(self, units, conv1d_filters, conv1d_kernel_size):
        super().__init__(units)
        self.location_layer = LocationSensitiveAttention.LocationLayer(units, conv1d_filters, conv1d_kernel_size)

    def call(self, query, values, previous_attention_weights):
        query = tf.expand_dims(query, 1)
        score = self.V(tf.math.tanh(self.memory_dense(values) + self.query_dense(query) + self.location_layer(previous_attention_weights)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class MyModel(tf.keras.Model):

    class DecoderCell(Layer):

        def __init__(self, units):
            super().__init__()
            self.attention = LocationSensitiveAttention(10, 32, 1)
            self.rnn_cell = LSTMCell(units)

            self.state_size = self.rnn_cell.state_size
            self.state_size.append(tf.TensorShape((length, 1)))
            self.dense = Dense(1, activation=tf.nn.tanh)

            self.output_size = self.rnn_cell.output_size + length

        def call(self, inputs, states, constants):
            encoder_outputs = constants[0]
            c, h, previous_attention_weights = states
            context_vector, attention_weights = self.attention(h, encoder_outputs, previous_attention_weights)
            x , new_states = self.rnn_cell(context_vector, states)
            x = self.dense(x)
            output = tf.concat((x, tf.reduce_sum(attention_weights, axis=-1)), axis=-1)
            new_states.append(attention_weights)
            return output, new_states

    def __init__(self):
        super().__init__()
        self.encoder = LSTM(20, return_sequences=True)
        self.decoder = RNN(MyModel.DecoderCell(20), return_sequences=True)

    def call(self, inputs):
        x = self.encoder(inputs)
        timesteps = inputs * 0
        timesteps = tf.reduce_sum(timesteps, axis=-1)
        timesteps = tf.linalg.matmul(timesteps, tf.zeros((timesteps.shape[-1], output_length)))
        timesteps = tf.expand_dims(timesteps, axis=-1)
        x = self.decoder(timesteps, constants=[x])
        output = x[:,:,:-length]
        attention_weights = x[:,:,-length:]
        
        attention_img = attention_weights[0] # (out_lenth lenth)
        tf.print(attention_img, output_stream='file:///home/hujk17/Attention_Test_TF2.x/attention_img_values.txt', summarize=-1)
        max_value_every_output_step_attention_img = tf.reduce_max(attention_img, axis=-1)
        tf.print(max_value_every_output_step_attention_img, output_stream='file:///home/hujk17/Attention_Test_TF2.x/max_value_every_output_step_attention_img.txt', summarize=-1)
        max_pos_every_output_step_attention_img = tf.argmax(attention_img, axis=-1)
        tf.print(max_pos_every_output_step_attention_img, output_stream='file:///home/hujk17/Attention_Test_TF2.x/max_pos_every_output_step_attention_img.txt', summarize=-1)
        
        attention_img_T = tf.transpose(attention_img) # (lenth out_lenth)
        max_value_every_input_step_attention_img = tf.reduce_max(attention_img_T, axis=-1)
        tf.print(max_value_every_input_step_attention_img, output_stream='file:///home/hujk17/Attention_Test_TF2.x/max_value_every_input_step_attention_img.txt', summarize=-1)
        max_pos_every_input_step_attention_img = tf.argmax(attention_img, axis=-1)
        tf.print(max_pos_every_input_step_attention_img, output_stream='file:///home/hujk17/Attention_Test_TF2.x/max_pos_every_input_step_attention_img.txt', summarize=-1)
        return output

model = MyModel()
optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['MeanSquaredError'])
model.build(input_shape=(None, length, 1))
model.summary()
model.fit(dataset, epochs=10000, shuffle=True)
