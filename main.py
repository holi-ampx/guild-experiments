"""
(activated virtualenv)
trying to install guild by requirements: git+https://github.com/guildai/guildai.git@master
says: needs six if used git+ in requirements...
After installing six, thep roblem persists.

Cloned guild ai from github (master at b362c51cfb7c970b01ae3d3585b61f6a44f99e61) and python setup.py install:
npm is not installed: [Errno 2] No such file or directory: 'npm': 'npm'
 -> https://askubuntu.com/questions/1088662/npm-depends-node-gyp-0-10-9-but-it-is-not-going-to-be-installed

python setup.py install
guild check
ModuleNotFoundError: No module named 'click'
pip install click

guild check
guild_version:             0.7.0.rc9
 ... ok!

(in this directory and this example:) guild run main
--- works, ok

python mainwithipy.py guild-ipy --toprint=20
"guild.ipy requires pandas - install it first before using "
installing pandas

python mainwithipy.py guild-ipy --toprint=20
guild.ipy.RunError: (<guild.run.Run '7697295420ed46f89393f696830e0768'>, FileNotFoundError(2, 'No such file or directory'))
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

print(tf.__version__)

XTEST = 10

def base_train(toprint=10):
    print(toprint)
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0

    test_images = test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10, verbose=1)

if __name__ == '__main__':
    base_train(toprint=XTEST)