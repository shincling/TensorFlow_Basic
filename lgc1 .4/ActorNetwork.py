from keras.layers import Dense, Flatten, Merge, Activation,Dropout,BatchNormalization,Input,Embedding,LSTM,Bidirectional,TimeDistributed
from keras.models import Sequential,Model
from keras import initializers
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

class ActorNetwork(object):
    def __init__(self,sess,state_size,action_size,BATCH_SIZE,TAU,LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        # self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.optimize = tf.train.AdadeltaOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states,action_grads):
        self.sess.run(self.optimize,feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    # def create_actor_network(self,state_size,fenli_size,action_size):
    def create_actor_network(self, state_size, action_size):
        print('now we build the actor model')
        # left_branch = Sequential()
        # left_branch.add(Dense(300,input_dim=state_size,activation='relu'))
        #
        # right_branch = Sequential()
        # right_branch.add(Dense(300,input_dim=fenli_size,activation='relu'))
        #
        # merged = Merge([left_branch,right_branch],mode='concat')
        #
        # model = Sequential()
        # model.add(merged)
        # model.add(Dense(300,activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        # model.add(Dense(action_size,activation='softmax'))
        # adam = Adam(lr=self.LEARNING_RATE)
        # model.compile(loss='mse',optimizer=adam)

        # model = Sequential()
        # model.add(Dense(600, input_dim=state_size, init='uniform'))
        # model.add(Activation('sigmoid'))
        # model.add(Dense(300,init='uniform'))
        # model.add(Activation('relu'))
        # model.add(Dense(action_size, init='uniform'))
        # model.add(Activation('sigmoid'))

        # S = Input(shape=[state_size])
        # h0 = Dense(16, init='uniform', activation='tanh')(S)
        # h1 = Dense(16, init='uniform', activation='tanh')(h0)
        # h2 = Dense(16, init='uniform', activation='tanh')(h1)
        # V = Dense(action_size,activation='sigmoid')(h2)
        # model = Model(input=S,output=V)

        # S = Input(shape=[state_size])
        # # x = Embedding(input_dim=state_size,output_dim=1024,input_length=257,init='normal')(S)
        # x1 = LSTM(output_dim=512,input_shape=(4,257), activation='tanh')(S)
        # x2 = LSTM(output_dim=512,activation='sigmoid',init='normal')(x1)
        # x3 = Dense(action_size,activation='sigmoid')(x2)
        # model = Model(input=S, output=x3)

        S = Input(shape=[3, state_size//3])
        bl1 = Bidirectional(LSTM(100, return_sequences=True, input_shape=(3, 257), kernel_initializer='lecun_uniform'))(S)
        # kernel_initializer='random_uniform'
        bl2 = Bidirectional(LSTM(100, return_sequences=False, input_shape=(3, 257), kernel_initializer='lecun_uniform'))(bl1)
        # bl2 = BatchNormalization()(bl2)
        output_mask_layer = Dense(action_size, kernel_initializer='lecun_uniform')(bl2)
        # kernel_initializer=initializers.random_normal(stddev=0.05)
        # output_mask_layer = BatchNormalization()(output_mask_layer)
        output_mask_layer = Activation('sigmoid')(output_mask_layer)
        model = Model(input=S, output=output_mask_layer)

        return model, model.trainable_weights, S
