import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow_datasets as tfds
import tensorflow as tf

from layers.blocks import transformer
from layers.custom_schedule import CustomSchedule

MAX_LENGTH = 40

def init_model(VOCAB_SIZE):
    tf.keras.backend.clear_session()

    # 하이퍼파라미터
    D_MODEL = 256
    NUM_LAYERS = 7
    NUM_HEADS = 8
    DFF = 512
    DROPOUT = 0.1

    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        dff=DFF,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)
    
    learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    return model

def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

def accuracy(y_true, y_pred):
    # 레이블의 크기는 (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def main():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
    train_data = pd.read_csv('ChatBotData.csv')
    print('챗봇 샘플의 개수 :', len(train_data))
    print(train_data.isnull().sum())

    questions = []
    answers = []

    for sentence in train_data['Q']:
        # 구두점에 대해서 띄어쓰기
        # ex) 12시 땡! -> 12시 땡 !
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = sentence.strip()
        questions.append(sentence)

    for sentence in train_data['A']:
        # 구두점에 대해서 띄어쓰기
        # ex) 12시 땡! -> 12시 땡 !
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = sentence.strip()
        answers.append(sentence)

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)

    # 시작 토큰과 종료 토큰에 대한 정수 부여.
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    # 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
    VOCAB_SIZE = tokenizer.vocab_size + 2

    print('시작 토큰 번호 :',START_TOKEN)
    print('종료 토큰 번호 :',END_TOKEN)
    print('단어 집합의 크기 :',VOCAB_SIZE)


    # 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(questions, answers):
        # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

        tokenized_inputs.append(sentence1)
        tokenized_outputs.append(sentence2)

    # 패딩
    questions = tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    answers = tf.keras.preprocessing.sequence.pad_sequences(tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    BATCH_SIZE = 64
    BUFFER_SIZE = 20000

    # 디코더의 실제값 시퀀스에서는 시작 토큰을 제거해야 한다.
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions,
            'dec_inputs': answers[:, :-1] # 디코더의 입력. 마지막 패딩 토큰이 제거된다.
        },
        {
            'outputs': answers[:, 1:]  # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다.
        },
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = init_model(VOCAB_SIZE)

    EPOCHS = 50
    model.fit(dataset, epochs=EPOCHS)
    model.save("models/")


if __name__ == "__main__":
    main()