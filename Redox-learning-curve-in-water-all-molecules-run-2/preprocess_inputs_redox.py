import numpy as np
import pandas as pd
import tensorflow as tf
import nfp

from tqdm import tqdm
from rdkit.Chem import MolFromSmiles, AddHs
tqdm.pandas()

from preprocessor import preprocessor

redox_df = pd.read_csv('/projects/rlmolecule/pstjohn/spin_gnn/redox_data.csv.gz')

if __name__ == '__main__':

#     redox_df = redox_df.sample(frac=1., random_state=1)
    
    data = np.load('redox_split.npz', allow_pickle=True)
    redf_train = redox_df[redox_df.smiles.isin(np.concatenate([data['train']]))]
    redf_valid = redox_df[redox_df.smiles.isin(np.concatenate([data['valid']]))]
    
    # split off 5000 each for test and valid sets
#     test, valid, train = np.split(redox_df.smiles.values, [1000, 2000])

    # Save these splits for later
#     np.savez_compressed('redox_split.npz', train=train, valid=valid, test=test)

#     redf_train = redox_df[redox_df.smiles.isin(train)]
#     redf_valid = redox_df[redox_df.smiles.isin(valid)]

    def inputs_generator(df, train=True):

        for _, row in tqdm(df.iterrows()):
            input_dict = preprocessor.construct_feature_matrices(row.smiles, train=train)
            input_dict['redox'] = row[['ionization energy', 'electron affinity']].values.astype(float)
            
            features = {key: nfp.serialize_value(val) for key, val in input_dict.items()}
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))

            yield example_proto.SerializeToString()

    serialized_train_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(redf_train, train=False),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords_redox/train.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_train_dataset)

    serialized_valid_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(redf_valid, train=False),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords_redox/valid.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_valid_dataset)

#    preprocessor.to_json('tfrecords_redf/preprocessor.json')
