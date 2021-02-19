import os
import sys
import math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
print(f"tensorflow version: {tf.__version__}")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import nfp

from preprocessor import preprocessor

from loss import AtomInfMask, KLWithLogits, RedoxAttention
from model import build_embedding_model


def parse_example(example):
    parsed = tf.io.parse_single_example(example, features={
        **preprocessor.tfrecord_features,
        **{'redox': tf.io.FixedLenFeature([], dtype=tf.string)}})

    # All of the array preprocessor features are serialized integer arrays
    for key, val in preprocessor.tfrecord_features.items():
        if val.dtype == tf.string:
            parsed[key] = tf.io.parse_tensor(
                parsed[key], out_type=preprocessor.output_types[key])
    
    # Pop out the prediction target from the stored dictionary as a seperate dict
    parsed['redox'] = tf.io.parse_tensor(parsed['redox'], out_type=tf.float64)
    redox = parsed.pop('redox')        
    return parsed, redox


model_name = "20210216_redox_lc"
inputs_dir = f"inputs/{model_name}"
outputs_dir = f"outputs/{model_name}"
tf_inputs_dir = f"{inputs_dir}/tfrecords"

splits = np.load(f'{inputs_dir}/split_redox.npz', allow_pickle=True)
print(splits)
num_train = len(splits['train'])
num_new_train = len(splits['train_new'])

# to build a learning curve, retrain the model using more and more data
runid = int(sys.argv[1])
print(f"runid: {runid}")
# split the 3K new calculations into groups evenly spaced by log
learn_curve_num_train_list = np.logspace(2, np.log10(num_new_train), num=6, dtype=int)
# make sure the last entry includes all training data
learn_curve_num_train_list[-1] = num_new_train
learn_curve_num_train = learn_curve_num_train_list[runid]
print(f"learning with {learn_curve_num_train} new training examples")

max_atoms = 80
max_bonds = 100
batch_size = 128

# Here, we have to add the prediction target padding onto the input padding
padded_shapes = (preprocessor.padded_shapes(max_atoms=None, max_bonds=None), [2])
padding_values = (preprocessor.padding_values, tf.constant(np.nan, dtype=tf.float64))


train_dataset = tf.data.TFRecordDataset(f'{tf_inputs_dir}/train.tfrecord.gz', compression_type='GZIP')
# add the new training examples here
train_new_dataset = tf.data.TFRecordDataset(f'{tf_inputs_dir}/train_new.tfrecord.gz', compression_type='GZIP')\
    .take(learn_curve_num_train)

train_dataset = train_dataset.concatenate(train_new_dataset)\
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache().shuffle(buffer_size=num_train).repeat()\
    .padded_batch(batch_size=batch_size,
                  padded_shapes=padded_shapes,
                  padding_values=padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset = tf.data.TFRecordDataset(f'{tf_inputs_dir}/valid.tfrecord.gz', compression_type='GZIP')\
    .map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache().shuffle(buffer_size=5000).repeat()\
    .padded_batch(batch_size=batch_size,
                  padded_shapes=padded_shapes,
                  padding_values=padding_values)\
    .prefetch(tf.data.experimental.AUTOTUNE)


input_tensors, atom_states, bond_states, global_states = build_embedding_model(
    preprocessor,
    dropout=0.0,
    atom_features=128,
    num_messages=6,
    num_heads=8,
    name='atom_embedding_model')

atom_class, bond_class, connectivity, n_atom = input_tensors
output = layers.Dense(2)(global_states[-1])

redox_model = tf.keras.Model(input_tensors, output)

if __name__ == "__main__":

    learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(1E-4, 1, 1E-5)
    weight_decay  = tf.keras.optimizers.schedules.InverseTimeDecay(1E-6, 1, 1E-5)
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    
    redox_model.compile(
        loss=nfp.masked_mean_absolute_error,
        optimizer=optimizer)
    
    redox_model.summary()

    lc_model_name = f'{outputs_dir}/n_{learn_curve_num_train}'
    print(model_name)
    os.makedirs(lc_model_name, exist_ok=True)

    filepath = lc_model_name + "/best_model.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, verbose=0)
    csv_logger = tf.keras.callbacks.CSVLogger(lc_model_name + '/log.csv')

    redox_model.fit(train_dataset,
                    validation_data=valid_dataset,
                    steps_per_epoch=math.ceil(num_train/batch_size),
                    validation_steps=math.ceil(5000/batch_size),
                    epochs=500,
                    callbacks=[checkpoint, csv_logger],
                    verbose=1)

    print("Finished")
