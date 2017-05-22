# coding:utf-8
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, merge,Activation,Input,Bidirectional,TimeDistributed,LSTM, BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        self.model,self.action,self.state = self.create_critic_network(state_size, action_size)
        self.target_model,self.target_action,self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output,self.action) #output应该是输出：也是Bs×257这么大的，注意,output或者action都知识一个tensor，所以这个语句最后的输出是一个只包含一个元素的列表（）
        self.sess.run(tf.global_variables_initializer())

    def gradients(self,states,actions):
        return self.sess.run(self.action_grads,feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1-self.TAU)*critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    # def create_critic_network(self,state_size,fenli_size,action_size):
    def create_critic_network(self, state_size , action_size):
        print('now we build the critic model')
        # s = Input(shape=[state_size])
        # a = Input(shape=[action_size])
        # left_branch = Sequential()
        # left_branch.add(Dense(600,input_dim=state_size ,activation='sigmoid'))
        #
        # right_branch = Sequential()
        # right_branch.add(Dense(600,input_dim=action_size,activation='sigmoid'))
        #
        # merged = Merge([left_branch,right_branch],mode='concat')
        #
        # model = Sequential()
        # model.add(merged)
        # model.add(Dense(600,activation='relu'))
        # model.add(Dense(300,activation='relu'))
        # model.add(Dense(action_size,activation='linear'))
        # adam = Adam(lr=self.LEARNING_RATE)
        # model.compile(loss='mse',optimizer=adam)
        # return model,a,s

        # S = Input(shape=[state_size])
        # A = Input(shape=[action_size], name='action2')
        # s1 = Dense(16, init='uniform', activation='linear')(S)
        # a1 = Dense(16, init='uniform', activation='linear')(A)
        # s2 = Dense(16, init='uniform', activation='tanh')(s1)
        # h1 = merge([s2, a1], mode='concat')
        # h2 = Dense(16, activation='tanh')(h1)
        # V = Dense(1, init='uniform', activation='relu')(h2)
        # model = Model(input=[S, A], output=V)
        # adam = Adam(lr=self.LEARNING_RATE)
        # model.compile(loss='mse', optimizer=adam)

        S = Input(shape=[3,257])
        A = Input(shape=[action_size], name='action')
        lb1 = Bidirectional(LSTM(100,return_sequences=False, input_shape=(3,257)))(S)
        s1 = Dense(300, kernel_initializer='random_uniform')(lb1)
        s1 = BatchNormalization()(s1)
        s1 = Activation('tanh')(s1)
        a1 = Dense(300, kernel_initializer='random_uniform')(A)
        a1 = BatchNormalization()(a1)
        a1 = Activation('tanh')(a1)
        m = merge([s1, a1], mode='concat')
        V = Dense(action_size, kernel_initializer='uniform')(m)
        # V = BatchNormalization()(V)
        V = Activation('relu')(V)
        model = Model(input=[S, A], output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)

        return model, A, S