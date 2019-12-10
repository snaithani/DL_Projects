import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime


def get_data():
    word_2_idx = {}
    tag_2_idx = {}
    words = []
    tags = []
    word_idx = 1
    tag_idx = 1
    
    curr_words = []
    curr_tags = []
    for line in open('data/ner.txt'):
        line = line.strip()
        if line:
            word, tag = line.split()
            word = word.lower()
            if word not in word_2_idx:
                word_2_idx[word] = word_idx
                word_idx += 1
            curr_words.append(word_2_idx[word])
            if tag not in tag_2_idx:
                tag_2_idx[tag] = tag_idx
                tag_idx += 1
            curr_tags.append(tag_2_idx[tag])
        else:
            words.append(curr_words)
            tags.append(curr_tags)
            curr_words = []
            curr_tags = []
            
    return words, tags, word_2_idx, tag_2_idx


def accuracy(y_pred, y_test):
    return np.mean(y_pred == y_test)


class NER(object):
    def __init__(self):
        tf.reset_default_graph()
        self.session = tf.InteractiveSession()
        self.tf_X = None
        self.tf_y = None
        self.predictions = None
        
    def fit(self, X, y, lr, beta1, beta2, num_epochs, V, K, emb_dims, h_dims,
            batch_sz = 64):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                              test_size = 0.25)
        N = len(X_train)
        
        t_len = max(len(x) for x in X_train)
        v_len = max(len(x) for x in X_valid)
        X_train = pad_sequences(X_train, t_len)
        y_train = pad_sequences(y_train, t_len)
        X_valid = pad_sequences(X_valid, v_len)
        y_valid = pad_sequences(y_valid, v_len)
        
        tf_X = tf.placeholder(tf.int32, (None, None))
        tf_y = tf.placeholder(tf.int32, (None, None))
        n_batches, seq_len = tf.shape(tf_X)[0], tf.shape(tf_X)[1]
        self.tf_X = tf_X
        self.tf_y = tf_y
        
        We = np.random.randn(V, emb_dims)
        Wo = np.random.randn(2 * h_dims, K)
        bo = np.zeros(K)
        
        tf_We = tf.Variable(We, dtype = tf.float32)
        tf_Wo = tf.Variable(Wo, dtype = tf.float32)
        tf_bo = tf.Variable(bo, dtype = tf.float32)
        
        emb = tf.nn.embedding_lookup(tf_We, tf_X)
        cell_fw = LSTMCell(h_dims, activation = tf.nn.relu)
        cell_bw = LSTMCell(h_dims, activation = tf.nn.relu)
        (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(
                                        cell_fw, cell_bw, emb, 
                                        dtype = tf.float32)
        output = tf.concat((output_fw, output_bw), 2)
        output = tf.reshape(output, (n_batches * seq_len, 2 * h_dims))
        logits = tf.matmul(output, tf_Wo) + tf_bo
        labels = tf.reshape(tf_y, [-1])
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
               labels = labels, logits = logits))
        
        train_op = tf.train.AdamOptimizer(learning_rate = lr, beta1 = beta1,
                                          beta2 = beta2).minimize(cost)
        
        predictions = tf.argmax(logits, axis = 1)
        predictions = tf.reshape(predictions, (n_batches, seq_len))
        self.predictions = predictions
        
        init = tf.global_variables_initializer()
        self.session.run(init)
        
        train_costs = []
        valid_costs = []
        num_batches = N // batch_sz
        for epoch in range(num_epochs):
            X_train, y_train = shuffle(X_train, y_train)
            train_cost = 0
            valid_cost = 0
            t0 = datetime.now()
            for i in range(num_batches):
                X_batch = X_train[i * batch_sz: (i + 1) * batch_sz]
                y_batch = y_train[i * batch_sz: (i + 1) * batch_sz]
                b_len = max(len(x) for x in X_batch)
                X_batch = pad_sequences(X_batch, b_len)
                y_batch = pad_sequences(y_batch, b_len)
                self.session.run(train_op, feed_dict = {tf_X: X_batch, 
                                                        tf_y: y_batch})
                if i % 100 == 0:
                    t_c, t_p = self.session.run((cost, predictions), 
                                                feed_dict = {tf_X: X_train,
                                                             tf_y: y_train})
                    v_c, v_p = self.session.run((cost, predictions),
                                                feed_dict = {tf_X: X_valid,
                                                             tf_y: y_valid})
                    train_cost += t_c
                    valid_cost += v_c
                    train_acc = accuracy(y_train, t_p)
                    valid_acc = accuracy(y_valid, v_p)
                    print('train cost is %f, train accuracy is %f'%(
                            t_c, train_acc))
                    print('valid cost is %f, valid accuracy is %f'%(
                            v_c, valid_acc))
            train_costs.append(train_cost)
            valid_costs.append(valid_cost)
            print('epoch %d completed in %s'%(i, datetime.now() - t0))
            
        plt.plot(train_costs)
        plt.plot(valid_costs)
        plt.show()
        
    def predict(self, X):
        y_pred = self.session.run(self.predictions, feed_dict = {self.tf_X: X})
        return y_pred
                    
        
if __name__ == '__main__':
    X, y, word_2_idx, tag_2_idx = get_data()
    V, K = len(word_2_idx) + 2, len(tag_2_idx) + 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    
    ner = NER()
    ner.fit(X_train, y_train, 0.001, 0.95, 0.95, 20, V, K, 50, 50)
    
    t_len = max(len(x) for x in X_test)
    X_test = pad_sequences(X_test, t_len)
    y_test = pad_sequences(y_test, t_len)
    y_pred = ner.predict(X_test)
    print(accuracy(y_pred, y_test))