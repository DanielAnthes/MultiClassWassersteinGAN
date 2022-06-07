#%%
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


def load_data(batchsize):
    (ds_train_im, ds_train_label), (ds_val_im, ds_val_label) = tf.keras.datasets.mnist.load_data()

    ds_train_im = np.expand_dims(ds_train_im.astype('float32') / 255, 3)
    ds_val_im = np.expand_dims(ds_val_im.astype('float32') / 255, 3)


    ds_train = tf.data.Dataset.from_tensor_slices((ds_train_im, ds_train_label))
    ds_train = ds_train.shuffle(60000)

    ds_val = tf.data.Dataset.from_tensor_slices((ds_val_im, ds_val_label))

    ds_train = ds_train.batch(batchsize)
    ds_val = ds_val.batch(batchsize)

    return ds_train, ds_val


def batch_by_class(batchsize, data, labels):
    datasets = []
    classes = np.unique(labels)
    for c in classes:
        data_class = data[labels == c]
        ds_class = tf.data.Dataset.from_tensor_slices(data_class)
        ds_class = ds_class.batch(batchsize)
        ds_class = ds_class.map(lambda x: (x,c))
        datasets.append(ds_class)
    ds = datasets[0]
    for d in datasets[1:]:
        ds = ds.concatenate(d)

    return ds


def load_data_by_class(batchsize):
    (ds_train_im, ds_train_label), (ds_val_im, ds_val_label) = tf.keras.datasets.mnist.load_data()
    ds_train_im = np.expand_dims(ds_train_im.astype('float32') / 255, 3)
    ds_val_im = np.expand_dims(ds_val_im.astype('float32') / 255, 3)

    ds_train_label = ds_train_label.astype('int32')
    ds_val_label = ds_val_label.astype('int32')

    ds_train = batch_by_class(batchsize, ds_train_im, ds_train_label)
    ds_val = batch_by_class(batchsize, ds_val_im, ds_val_label)

    return ds_train, ds_val



