from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from IPython.display import SVG
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from keras_self_attention import SeqSelfAttention

fuckrasppi = load_model('mnist1.h5')
plot_model(fuckrasppi, to_file='./fuckrasppi.png')