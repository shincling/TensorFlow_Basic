# coding:utf-8

import numpy as np
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import tensorflow as tf
# from mat2python import load_matrix
from keras import backend as K
from keras.models import model_from_json
from keras.models import Sequential
from keras.optimizers import  Adam
# from keras.engine.training import collect_trainable_weights
# from t_separate import separate
import json
from env import first_state
from env import separate
from env import pro_wav
import argparse
import h5py


OU = OU()

def speech_separate(train_indicator=0):         #train_indicator = 0 means simply run ,1 means train
    BUFFER_SIZE = 1000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001
    LRA = 0.0001
    LRC = 0.0001

    action_size = 257
    state_size = 771


    EXPLORE = 10000
    # episode_count = 2000
    if train_indicator == 0:   #only separate
        episode_count = 1
    else:
        episode_count = 2

    max_steps = 10000
    done = False
    step = 0
    epsilon = 1


    #tensorflow gpu optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # sess = tf.Session()

    K.set_session(sess)

    actor = ActorNetwork(sess,state_size,action_size,BATCH_SIZE,TAU,LRA)
    critic = CriticNetwork(sess, state_size, action_size, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)

    if train_indicator == 0:
        print('now we load the weight in ddpg')
        try:
            actor.model.load_weights('actormodel.h5')
            critic.model.load_weights('criticmodel.h5')
            actor.target_model.load_weights('actormodel.h5')
            critic.target_model.load_weights('criticmodel.h5')
            print('weight load successfully')
        except:
            print('cannot find the weight')

    if train_indicator == 1:
        print('now we load the preparing weight of actor in ddpg')
        try:
            actor.model.load_weights('pre_actormodel.h5')
            actor.target_model.load_weights('pre_actormodel.h5')
            print('weight load successfully')
        except:
            print('cannot find the weight')

    print('start to separate the speech')

    for i in range(episode_count):

        print('episode:' + str(i) + 'replay buffer' + str(buff.count()))

        total_reward = 0.

        # 记载初始状态
        s_t = first_state

        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_size])
            noise_t = np.zeros([1,action_size])

            # 根据当前的状态预测动作
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0], s_t.shape[1]))

            # noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
            # noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
            # noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

            # noise_t[0] = [train_indicator * max(epsilon, 0) * OU.function(a_t_o, 0.5, 1, 0.6) for a_t_o in a_t_original[0]]
            # if the design is such,every action'change is the same
            # a_t[0] = a_t_original[0] + noise_t[0]

            # 对动作值二值化
            a_t[0] = a_t_original[0]
            for m in range(a_t[0].shape[0]):
                if a_t[0][m] >= 1:
                    a_t[0][m] = 1
                if a_t[0][m] < 0:
                    a_t[0][m] = 0

            # a_t[0] = a_t_original
            # 根据动作（mask），去分离语音，得到(下一个状态，奖励，是否完成整条语音分离)
            s_t1,r_t,done = separate(a_t[0], j)

            # 把这些数据保存到buffer中，然后随机提取一批，目的是减少样本之间的相关性
            buff.add(s_t, a_t[0], r_t, s_t1, done)
            batch = buff.getbatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            # y_t = []

            # 根据新的状态预测动作，然后在critc中输入（新状态，新新动作），产生Q  TODO-shin:这块critic的model的输入是俩，后面可能得结合AC算法原文搞一下
            target_q_values = critic.target_model.predict([new_states,(actor.target_model.predict(new_states))])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                    # y_t.append(rewards[k])
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
                    # y_t.append(rewards[k] + GAMMA*target_q_values[k])

            # y_t = np.resize(y_t, [len(y_t), 1])

            # 这部分是进行网络更新的
            if (train_indicator):
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states,grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            # a_t[0,0:5]是输出当前动作的前五位数值
            print("Episode", i, "Step", step, "Action", a_t[0,0:5], "Reward", r_t, "Loss", loss)

            step += 1

            # 如果整条的语句分离完成了，则测试整条语音的分离情况
            if done:
                pro_wav()
                break

        # 保存模型参数
        if np.mod(i,2) == 0:
            if (train_indicator):
                print('now we save model')
                actor.model.save_weights('actormodel.h5',overwrite=True)
                with open('actormodel.json','w') as outfile:
                    json.dump(actor.model.to_json(),outfile)

                critic.model.save_weights('criticmodel.h5',overwrite=True)
                with open('criticmodel.json','w') as outfile:
                    json.dump(critic.model.to_json(),outfile)

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print('finish')

if __name__ == '__main__':
    speech_separate(train_indicator=1)

