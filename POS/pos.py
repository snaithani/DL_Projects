import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.nn import dynamic_rnn
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime


def get_data():
    word_2_idx = {}
    tag_2_idx = {}
    words_train = []
    tags_train = []
    words_test = []
    tags_test = []
    word_idx = 1
    tag_idx = 1
    
    curr_words = []
    curr_tags = []
    for line in open('chunking/train.txt'):
        line = line.strip()
        if line:
            word, tag, _ = line.split()
            if word not in word_2_idx:
                word_2_idx[word] = word_idx
                word_idx += 1
            curr_words.append(word_2_idx[word])
            if tag not in tag_2_idx:
                tag_2_idx[tag] = tag_idx
                tag_idx += 1
            curr_tags.append(tag_2_idx[tag])
        else:
            words_train.append(curr_words)
            tags_train.append(curr_tags)
            curr_words = []
            curr_tags = []
            
    curr_words = []
    curr_tags = []
    for line in open('chunking/test.txt'):
        line = line.strip()
        if line:
            word, tag, _ = line.split()
            if word in word_2_idx:
                curr_words.append(word_2_idx[word])
            else:
                curr_words.append(word_idx)
            if tag in tag_2_idx:
                curr_tags.append(tag_2_idx[tag])
            else:
                curr_tags.append(tag_idx)
        else:
            words_test.append(curr_words)
            tags_test.append(curr_tags)
            curr_words = []
            curr_tags = []
            
    return (words_train, tags_train, words_test, tags_test, 
            word_2_idx, tag_2_idx)


def accuracy(y_pred, y_test):
    return np.mean(y_pred == y_test)


class POS(object):
    def __init__(self):
        tf.reset_default_graph()
        self.session = tf.InteractiveSession()
        self.tf_X = None
        self.tf_y = None
        self.predictions = None
        
    def fit(self, X, y, num_epochs, embedding_dims, V, K, hidden_dims, lr,
            beta1 = 0.95, beta2 = 0.95, batch_sz = 32):
        X_train, X_valid, y_train, y_valid = train_test_split(
                                             X, y, test_size = 0.25)
        len_t = max(len(x) for x in X_train)
        len_v = max(len(x) for x in X_valid)
        X_train = pad_sequences(X_train, len_t)
        y_train = pad_sequences(y_train, len_t)
        X_valid = pad_sequences(X_valid, len_v)
        y_valid = pad_sequences(y_valid, len_v)
        
        We = np.random.randn(V, embedding_dims)
        Wo = np.random.randn(hidden_dims, K)
        bo = np.zeros(K)
        N = X_train.shape[0]
        
        tf_X = tf.placeholder(tf.int32, (None, None))
        tf_y = tf.placeholder(tf.int32, (None, None))
        
        self.tf_X = tf_X
        self.tf_y = tf_y
        
        tf_We = tf.Variable(We, dtype = tf.float32)
        tf_Wo = tf.Variable(Wo, dtype = tf.float32)
        tf_bo = tf.Variable(bo, dtype = tf.float32)
        
        emb = tf.nn.embedding_lookup(tf_We, tf_X)
        cell = LSTMCell(hidden_dims, activation = tf.nn.relu)
        outputs, states = dynamic_rnn(cell, emb, dtype = tf.float32)
        outputs = tf.reshape(outputs, (tf.shape(tf_X)[0] * tf.shape(tf_X)[1],
                                       hidden_dims))
        
        logits = tf.matmul(outputs, tf_Wo) + tf_bo
        labels = tf.reshape(tf_y, [-1])
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                              labels = labels, logits = logits))
        train_op = tf.train.AdamOptimizer(learning_rate = lr, beta1 = beta1,
                                          beta2 = beta2).minimize(cost)
        
        predictions = tf.argmax(logits, 1)
        predictions = tf.reshape(predictions, (tf.shape(tf_X)[0], 
                                               tf.shape(tf_X)[1]))
        
        self.predictions = predictions
        
        num_batches = N // batch_sz
        t_costs = []
        v_costs = []
        init = tf.global_variables_initializer()
        self.session.run(init)
        
        for epoch in range(num_epochs):
            t0 = datetime.now()
            X_train, y_train = shuffle(X_train, y_train)
            t_cost = 0
            v_cost = 0
            for i in range(num_batches):
                X_batch = X_train[i * batch_sz: (i + 1) * batch_sz]
                y_batch = y_train[i * batch_sz: (i + 1) * batch_sz]
                len_b = max(len(x) for x in X_batch)
                X_batch = pad_sequences(X_batch, len_b)
                y_batch = pad_sequences(y_batch, len_b)
                self.session.run(train_op, feed_dict = {tf_X: X_batch,
                                                        tf_y: y_batch})
                if i % 100 == 0:
                    t_c, t_pred = self.session.run(
                        (cost, predictions),
                        feed_dict = {tf_X: X_train, tf_y: y_train})
                    t_cost += t_c
                    t_acc = accuracy(t_pred, y_train)
                    v_c, v_pred = self.session.run(
                        (cost, predictions), 
                        feed_dict = {tf_X: X_valid, tf_y: y_valid})
                    v_cost += v_c
                    v_acc = accuracy(v_pred, y_valid)
                    print('train cost: %f, train accuracy: %f'%(t_cost, t_acc))
                    print('valid cost: %f, valid accuracy: %f'%(v_cost, v_acc))
            t_costs.append(t_cost)
            v_costs.append(v_cost)
            print('Epoch completed in %s'%(datetime.now() - t0))
        
        plt.plot(t_costs)
        plt.plot(v_costs)
        plt.show()
        
    def predict(self, X):
        y_pred = self.session.run(self.predictions, feed_dict = {self.tf_X: X})
        return y_pred
    
def main():
    X_train, y_train, X_test, y_test, word_2_idx, tag_2_idx = get_data()
    V = len(word_2_idx) + 2
    K = len(tag_2_idx) + 2

    pos = POS()
    pos.fit(X_train, y_train, 5, 20, V, K, 20, 0.01)
    
    y_pred = pos.predict(X_test)
    print(accuracy(y_pred, y_test))
    
if __name__ == '__main__':
    main()