import os
import numpy as np
import pandas as pd
import tensorflow as tf
import nfp

from tqdm import tqdm
from rdkit.Chem import MolFromSmiles, AddHs
tqdm.pandas()
from preprocessor import preprocessor


redox_df = pd.read_csv('/projects/rlmolecule/pstjohn/spin_gnn/redox_data.csv.gz')
redox_new_calcs = pd.read_csv('/projects/rlmolecule/pstjohn/spin_gnn/20210216_fixed_rl_redox_data.csv')

model_name = "20210216_redox_lc"
inputs_dir = f"inputs/{model_name}"
tf_inputs_dir = f"{inputs_dir}/tfrecords"

if __name__ == '__main__':

    redox_df = redox_df.sample(frac=1., random_state=1)
    redox_new_calcs = redox_new_calcs.sample(frac=1., random_state=1)
    
    # split off 1000 each for test and valid sets
    test, valid, train = np.split(redox_df.smiles.values, [1000, 2000])

    # split off 500 each for test
    test_new, train_new = np.split(redox_new_calcs.smiles.values, [500])

    # Save these splits for later
    splits_file = f'{inputs_dir}/split_redox.npz'
    os.makedirs(os.path.dirname(splits_file), exist_ok=True)
    print(f"saving splits to {splits_file}")
    np.savez_compressed(splits_file, train=train, valid=valid, test=test, test_new=test_new, train_new=train_new)

    redf_train = redox_df[redox_df.smiles.isin(train)]
    redf_valid = redox_df[redox_df.smiles.isin(valid)]
    redf_new_train = redox_new_calcs[redox_new_calcs.smiles.isin(train_new)]    
    redf_new_test = redox_new_calcs[redox_new_calcs.smiles.isin(test_new)]    

    def inputs_generator(df, train=True):

        for _, row in tqdm(df.iterrows()):
            input_dict = preprocessor.construct_feature_matrices(row.smiles, train=train)
            input_dict['redox'] = row[['ionization energy', 'electron affinity']].values.astype(float)

            features = {key: nfp.serialize_value(val) for key, val in input_dict.items()}
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))

            yield example_proto.SerializeToString()

    inputs = [
            ('train_new', redf_new_train),
            ('test_new', redf_new_test),
            ('train', redf_train),
            ('valid', redf_valid)]

    for dataset_name, dataset in inputs:
        serialized_dataset = tf.data.Dataset.from_generator(
            lambda: inputs_generator(dataset, train=False),
            output_types=tf.string, output_shapes=())

        filename = f'{tf_inputs_dir}/{dataset_name}.tfrecord.gz'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"writing {filename}")
        writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
        writer.write(serialized_dataset)
