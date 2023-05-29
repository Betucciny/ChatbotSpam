import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

import re
import pandas as pd
import numpy as np

import email
import random
import matplotlib.pyplot as plt
import seaborn as sn

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords = stopwords.words('english')

Nsamp = 2000  # number of samples to generate in each class - 'spam', 'not spam'
maxtokens = 5000  # the maximum number of tokens per document
maxtokenlen = 500  # the maximum length of each token

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def tokenize(row):
    if row is None or row == '':
        tokens = ""
    else:
        try:
            tokens = row.split(" ")[:maxtokens]
        except:
            tokens = ""
    return tokens


def reg_expressions(row):
    tokens = []
    try:
        for token in row:
            token = token.lower()
            token = re.sub(r'[\W\d]', "", token)
            token = token[:maxtokenlen]  # truncate token
            tokens.append(token)
    except:
        token = ""
        tokens.append(token)
    return tokens


def stop_word_removal(row):
    token = [token for token in row if token not in stopwords]
    token = filter(None, token)
    return token


def extract_messages(df):
    messages = []
    for item in df["message"]:
        # Return a message object structure from a string
        e = email.message_from_string(item)
        # get message body
        message_body = e.get_payload()
        messages.append(message_body)
    print("Successfully retrieved message body from e-mails!")
    return messages


def build_model(max_seq_length):
    # tf hub bert model path
    bert_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)

    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer(bert_path, trainable=False)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]  # [batch_size, 768].

    # train dense classification layer on top of extracted pooled output features
    dense = tf.keras.layers.Dense(256, activation="relu")(pooled_output)
    pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

    model = tf.keras.Model(inputs=text_input, outputs=pred, name="spam_detector_bert")
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    model.summary()

    return model


def unison_shuffle(a, b):
    p = np.random.permutation(len(b))
    data = a[p]
    header = np.asarray(b)[p]
    return data, header


def convert_data(raw_data, header):
    converted_data, labels = [], []
    for i in range(raw_data.shape[0]):
        out = ' '.join(raw_data[i])
        converted_data.append(out)
        labels.append(header[i])
    converted_data = np.array(converted_data, dtype=object)[:, np.newaxis]
    return converted_data, np.array(labels)


def main():
    filepath = "spam/spamdata/enron-email-dataset/emails.csv"
    emails = pd.read_csv(filepath)
    print("Successfully loaded {} rows and {} columns!".format(emails.shape[0], emails.shape[1]))
    print(emails.head())
    print(emails.loc[0]["message"])

    bodies = extract_messages(emails)
    bodies_df = pd.DataFrame(random.sample(bodies, 10000))
    pd.set_option('display.max_colwidth', 300)
    print(bodies_df.head())

    filepath = "spam/spamdata/fraudulent-email-corpus/fradulent_emails.txt"
    with open(filepath, 'r', encoding="latin1") as file:
        data = file.read()

    fraud_emails = data.split("From r")
    print("Successfully loaded {} spam emails!".format(len(fraud_emails)))

    fraud_bodies = extract_messages(pd.DataFrame(fraud_emails, columns=["message"], dtype=str))
    fraud_bodies_df = pd.DataFrame(fraud_bodies[1:])
    fraud_bodies_df.head()  # you could do print(fraud_bodies_df.head()), but Jupyter displays this nicer for pandas DataFrames

    # Convert everything to lower-case, truncate to maxtokens and truncate each token to maxtokenlen
    EnronEmails = bodies_df.iloc[:, 0].apply(tokenize)
    EnronEmails = EnronEmails.apply(stop_word_removal)
    EnronEmails = EnronEmails.apply(reg_expressions)
    EnronEmails = EnronEmails.sample(Nsamp)

    SpamEmails = fraud_bodies_df.iloc[:, 0].apply(tokenize)
    SpamEmails = SpamEmails.apply(stop_word_removal)
    SpamEmails = SpamEmails.apply(reg_expressions)
    SpamEmails = SpamEmails.sample(Nsamp)

    raw_data = pd.concat([SpamEmails, EnronEmails], axis=0).values
    # %%
    print("Shape of combined data represented as numpy array is:")
    print(raw_data.shape)
    print("Data represented as numpy array is:")
    print(raw_data)

    # corresponding labels
    header = ([1] * Nsamp)
    header.extend(([0] * Nsamp))

    raw_data, header = unison_shuffle(raw_data, header)
    # split into independent 70% training and 30% testing sets
    idx = int(0.7 * raw_data.shape[0])
    # 70% of data for training
    train_x, train_y = convert_data(raw_data[:idx], header[:idx])
    # remaining 30% for testing
    test_x, test_y = convert_data(raw_data[idx:], header[idx:])

    print("train_x/train_y list details, to make sure it is of the right form:")
    print(len(train_x))
    print(train_x)
    print(train_y[:5])
    print(train_y.shape)

    model = build_model(maxtokens)

    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=5, batch_size=32)

    y_pred = model.predict(test_x)

    test = pd.DataFrame({
        'prob': y_pred[:, 0],
        'pred': np.where(y_pred[:, 0] > 0.5, "Spam", "No Spam"),
        'true': np.where(test_y == 1, "Spam", "No Spam")
    })
    confusion_matrix = pd.crosstab(test['pred'], test['true'], rownames=['Predicted'], colnames=['True'])
    plot = sn.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.show()
    fig = plot.get_figure()
    fig.savefig('BERTConfusionMatrix.pdf', format='pdf')
    fig.savefig('BERTConfusionMatrix.png', format='png')
    print(test.head(20))


    df_history = pd.DataFrame(history.history)
    fig, ax = plt.subplots()
    plt.plot(range(df_history.shape[0]), df_history['val_accuracy'], 'bs--', label='validation')
    plt.plot(range(df_history.shape[0]), df_history['accuracy'], 'r^--', label='training')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('BERT Email Classification Training')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    fig.savefig('BERTConvergence.pdf', format='pdf')
    fig.savefig('BERTConvergence.png', format='png')

    model.save('spamdetector.h5')


if __name__ == "__main__":
    main()
