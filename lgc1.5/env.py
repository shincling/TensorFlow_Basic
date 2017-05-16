# coding:utf-8

import matlab.engine
import scipy.io.wavfile as wav
import numpy as np
import soundfile
from scipy import signal
from librosa import stft, istft

#  和matlab进行交互，目的是调用matlab中的BSS_EVAL评价函数，返回4个指标
eng = matlab.engine.start_matlab()

# 傅利叶变换的点数，也是window_size的大小
F_S = 512
win_size = F_S

# 采用的汉明窗
WINDOWS=np.hamming(512)

# 读入语音数据
(female_read, fs) = soundfile.read("female_train.wav")
(male_read, fs) = soundfile.read("male_train.wav")
print(fs)                       #16000
print(female_read.shape[0])     #357994
print(male_read.shape[0])       #370283
if female_read != male_read:
    maxlength = max(female_read.shape[0],male_read.shape[0])
print(maxlength)               #370283
print(type(female_read))

# 因为男声和女生的语音数据的维度不同，进行补齐
zeros = np.zeros((maxlength-female_read.shape[0]))
# zeros = zeros + 0.000000000000001
female_original = np.concatenate((female_read, zeros))
male_original = male_read
print(female_original.shape[0])

# 归一化设置 (范围是-1到1 ，平方和为1）
male_original = male_original/np.sqrt(np.sum(male_original*male_original))
female_original = female_original/np.sqrt(np.sum(female_original*female_original))
mix = female_original + male_original #直接混合，线性相加

# 对混合的语音进行STFT短时傅里叶变换，输出的结果是矩阵（语谱图），形状大小是（257，帧数）
w = F_S
n = w
ov = w//2
h = w - ov
win = WINDOWS
c = 0
ncols = 1 + int((mix.size-n)/h)
d = np.zeros(((1+n//2), ncols))
# d = np.zeros((n, ncols))
for b in range(0, (int(mix.size)-n), h):
    u = win*mix[b:(b+n)]
    t = np.fft.fft(u)
    d[:, c] = t[0:(1+n//2)]
    # d[:, c] = t
    c += 1

print(c)
print(d.shape)

mixture = d.T

# first state ，设置环境的初始状态
s0 = mixture[0,:]
s1 = np.vstack((s0, s0))
first_state = np.vstack((s1, s0))

# 搜集所有帧的action(掩膜)
action_matrix = np.zeros(257)

# 保存分离后的每一帧语音的语谱
final = np.zeros(257)
out_wav = np.array([])

# 分离动作函数，返回的参数，（下一个状态，获得奖励，是否分离完成）
def separate(action, i):

    global final
    # 使用action（掩膜乘以混合语音，分离当前帧的语音）
    fenli = action * mixture[i, :]
    final = np.vstack((final, fenli))

    # 把每一帧的mask保存起来
    global action_matrix
    action_matrix = np.vstack((action_matrix,action))

    # 产生下一个状态，使用的三帧（前两帧+当前帧）
    if i == 0:
        new_state = np.concatenate((mixture[1], mixture[1], mixture[1]))
        new_state = new_state.reshape(3,(mixture[1].shape[0]))
    elif i == mixture.shape[0]-1:
        new_state = np.concatenate((final[i+1], final[i+1], final[i+1]))
        new_state = new_state.reshape(3,(mixture[1].shape[0]))
    else:
        new_state = np.concatenate((mixture[i-1], mixture[i], mixture[i+1]))
        new_state = new_state.reshape(3,(mixture[1].shape[0]))

    # 把分离了的语音进行IFFT变换，逆变换会语音数据
    fenli = np.concatenate((fenli, (fenli.conj())[-2:0:-1]), 0)
    wav_fenli = np.fft.ifft(fenli)

    # 之前成加了窗，这里要去除
    wav_fenli = abs(wav_fenli)/WINDOWS

    # BSS_EVAL评价函数的参数（纯净语音，单一的噪音，预测的纯净语音，预测的噪音，混合的语音）
    # 这里对应的是（目标女声，目标男声，预测的女声，预测的男声，原始男女混合声）
    wav_truth_signal = (female_original[i * 256:512 + i * 256]).tolist()
    wav_truth_noise = (male_original[i * 256:512 + i * 256]).tolist()
    wav_pred_signal = (wav_fenli).tolist()
    wav_pred_noise = (mix[i * 256:512 + i * 256] - wav_fenli).tolist()
    wav_mix = (mix[i * 256:512 + i * 256]).tolist()

    # 这一部分是进行数据的转化，变成matlab中的格式
    wav_truth_signal = matlab.double(wav_truth_signal)
    wav_truth_noise = matlab.double(wav_truth_noise)
    wav_pred_signal = matlab.double(wav_pred_signal)
    wav_pred_noise = matlab.double(wav_pred_noise)
    wav_mix = matlab.double(wav_mix)
    print ('meizhen yuci de xinhao')
    print(wav_pred_signal)

    params = eng.BSS_EVAL(wav_truth_signal, wav_truth_noise, wav_pred_signal, wav_pred_noise,wav_mix)
    print(params)
    print ('---------------------------------------------------------------------------------------')

    p1 = params['SIR']
    p2 = params['SDR']
    p3 = params['SAR']
    p4 = params['NSDR']

    # 根据三个指标，设置产生reward
    if (p1 > 0) and (p2 > 0) and (p3 > 0) :
        reward = 5
    elif (p1 < 0) and (p2 < 0) and (p3 < 0) :
        reward = -5
    elif (p2 > 0) and (p1 > 0) and (p3 < 0):
        reward = 2
    elif (p2 > 0) and (p3 > 0) and (p1 < 0):
        reward = 2
    elif (p4 > 0) and (p2 > 0):
        reward = 3
    elif (p4 < 0)and(p2 < 0):
        reward = -3
    elif (p3 < 0)and(p2< 0):
        reward = -1
    else:
        reward = 1

    global done
    done = False

    global  out_wav
    out_wav = final

    # 当一条语音分离完成时，done=True
    if (i == mixture.shape[0] - 1):
        final = np.zeros(257)
        done = True

        #  保存整条语音每一帧的mask
        np.savetxt('action_matrix.ods', action_matrix, delimiter=',')
        action_matrix = np.zeros(257)

    # # reward = (p2 + p1 + p3 + p4)*1

    return new_state, reward, done

# 最后再次评价整条语音，参看最后的分离指标
def pro_wav():

    w = out_wav[1:,:]
    for i in range(w.shape[0]):
        w[i] = w[i].conj()
    w = w.T
    print(w.shape)

    wavout_signal = istft(w, 256, 512, window='hamming',center=False)
    print (wavout_signal.shape)
    soundfile.write('final.wav', wavout_signal, 16000)

    wav_truth_signal = (female_original).tolist()
    wav_truth_noise = (male_original).tolist()
    wav_pred_signal = (wavout_signal).tolist()
    wav_pred_noise = (mix[0:wavout_signal.shape[0]]-wavout_signal).tolist()
    wav_mix = (mix).tolist()

    wav_truth_signal = matlab.double(wav_truth_signal)
    wav_truth_noise = matlab.double(wav_truth_noise)
    wav_pred_signal = matlab.double(wav_pred_signal)
    wav_pred_noise = matlab.double(wav_pred_noise)
    wav_mix = matlab.double(wav_mix)

    params = eng.BSS_EVAL(wav_truth_signal, wav_truth_noise, wav_pred_signal, wav_pred_noise, wav_mix)
    print('the whole speech -------------------------------')
    print(params)




