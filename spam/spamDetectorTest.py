import tensorflow as tf
from spamDetectorTrain import tokenize, reg_expressions, stop_word_removal, extract_messages
import pandas as pd
import numpy as np


def main():
    model = tf.keras.models.load_model("model.h5")


