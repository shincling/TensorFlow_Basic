# coding:utf-8

import matlab.engine
import scipy.io.wavfile as wav
import numpy as np
import soundfile
from scipy import signal
from librosa import stft, istft


eng = matlab.engine.start_matlab()

# 入读语音数据，并补齐较短的语音，让两条语音的长度一样
def data_load():
    female_train, fs = soundfile.read('female_train.wav')
    male_train, fs = soundfile.read('male_train.wav')

    error_len = female_train.shape[0] - male_train.shape[0]
    if error_len < 0:
        zeros = np.zeros(-error_len)
        female_train = np.concatenate((female_train, zeros))
    if error_len > 0:
        zeros = np.zeros(error_len)
        male_train = np.concatenate((male_train, zeros))

    female_train /= np.sqrt((female_train * female_train).sum())
    male_train /= np.sqrt((male_train * male_train).sum())


    return female_train, male_train

# 这部分对读入的语音进行stft变换，产生混合的频谱和目标mask(也就是action)
# noshift 是说没有使用shift操作来增加数据量
# 使用了工具箱里的STFT函数，不用以前那样繁琐的计算
def data_noshift(female_data, male_data):
    male_spec = stft(male_data, n_fft=512, hop_length=256, window='hamming', center=False)
    female_spec = stft(female_data, n_fft=512, hop_length=256, window='hamming', center=False)
    mix_data = female_data + male_data
    mix_spec = stft(mix_data, n_fft=512, hop_length=256, window='hamming', center=False)
    print(mix_spec.shape)

    # data_target = abs(female_spec) > abs(male_spec)
    # data_target = abs(female_spec/mix_spec)
    data_target = np.sqrt((np.power(abs(female_spec), 2))/(np.power(abs(female_spec), 2)+np.power(abs(male_spec), 2)))
    #TODO:这种mask的设置其实是存在理论上的误差的，将来可以考虑如何还原回去

    return mix_spec, data_target

female_train, male_train = data_load()
mix_speech = female_train + male_train
mix_spec, data_target = data_noshift(female_train, male_train)
print('mix_spec.shape', mix_spec.shape)
print('data_target', data_target.shape)

np.savetxt('data_target.ods', data_target, delimiter=',')

mix_spec = mix_spec.T
data_target = data_target.T
mask = data_target

# first state
s0 = mix_spec[0, :]
s1 = np.vstack((s0, s0))
first_state = np.vstack((s1, s0))

action_matrix = np.zeros(257)

# # done = False

out_wav = np.array([])

def separate(action, i):

    global final

    # reward = sum(-abs(pre_mask - data_target[i])) + 128

    # reward = sum(-abs(pre_mask - data_target[i]))

    # reward的设置，尽量让reward在[-1, 1]之间
    reward = sum(-abs(action - data_target[i]))

    print ('reward', reward)
    reward = 1+reward/128

    # 这部分操作是讲每一帧的action都保存下来，方便对比目标action
    global action_matrix
    action_matrix = np.vstack((action_matrix, action))

    # 这部分是分离语音后，产生下一个状态
    # print(new_state)
    if i == 0:
        # new_state = np.vstack((mixture[1], final[1], final[1]))
        # print(new_state)
        new_state = np.concatenate((mix_spec[0], mix_spec[1], mix_spec[1]))
        new_state = new_state.reshape(3,(mix_spec[1].shape[0]))
        # new_state = np.concatenate((mixture[1], mixture[1], mixture[1]))
    elif i == mix_spec.shape[0]-1:
        new_state = np.concatenate((mix_spec[i], mix_spec[i], mix_spec[i]))
        new_state = new_state.reshape(3,(mix_spec[1].shape[0]))
    else:
        new_state = np.concatenate((mix_spec[i-1], mix_spec[i], mix_spec[i+1]))
        new_state = new_state.reshape(3,(mix_spec[1].shape[0]))


    global done
    done = False

    global out_wav

    # 这部分是如果整条语音分离完毕，保存所有的action
    # if (i == (mix_spec.shape[0]-1)):
    if (i == 0):
        done = True

        global mask
        mask = action_matrix[1:, :]

        np.savetxt('action_matrix.ods', mask, delimiter=',')
        action_matrix = np.zeros(257)


    return new_state, reward, done



# 这部分是评价分离完的整条语音
def pro_wav():

    w = mask * mix_spec
    w = w.T
    wavout_signal = istft(w, 256, 512, window='hamming',center=False)
    print (wavout_signal.shape)
    soundfile.write('final.wav', wavout_signal, 16000)


    wav_truth_signal = (female_train).tolist()
    wav_truth_noise = (male_train).tolist()
    wav_pred_signal = (wavout_signal).tolist()
    wav_pred_noise = (mix_speech[0:wavout_signal.shape[0]]-wavout_signal).tolist()
    wav_mix = (mix_speech).tolist()

    wav_truth_signal = matlab.double(wav_truth_signal)
    wav_truth_noise = matlab.double(wav_truth_noise)
    wav_pred_signal = matlab.double(wav_pred_signal)
    wav_pred_noise = matlab.double(wav_pred_noise)
    wav_mix = matlab.double(wav_mix)

    params = eng.BSS_EVAL(wav_truth_signal, wav_truth_noise, wav_pred_signal, wav_pred_noise, wav_mix)
    print('the whole speech -------------------------------')
    print(params)




