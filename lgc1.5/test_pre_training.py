import numpy as np
import soundfile
from librosa import stft, istft
import keras
from keras.layers import Dense, Bidirectional, LSTM, Input
from keras.models import Model
from keras.optimizers import Adam
import matlab.engine
import json

eng = matlab.engine.start_matlab()

def create_actor_network(state_size, action_size):
    print('now we build the actor model')
    S = Input(shape=[3, state_size // 3])
    bl1 = Bidirectional(LSTM(100, return_sequences=True, input_shape=(3, 257)))(S)
    bl2 = Bidirectional(LSTM(100, return_sequences=False, input_shape=(3, 257)))(bl1)
    output_mask_layer = Dense(action_size, activation='sigmoid')(bl2)
    model = Model(input=S, output=output_mask_layer)
    adam = Adam(lr=0.001)
    model.compile(loss='mse', optimizer=adam)
    print ('we finish building the model')
    return model


(female_read, fs) = soundfile.read("female_train.wav")
(male_read, fs) = soundfile.read("male_train.wav")

print(fs)                       #16000
print(female_read.shape[0])     #357994
print(male_read.shape[0])       #370283
if female_read != male_read:
    maxlength = max(female_read.shape[0],male_read.shape[0])
print(maxlength)               #370283
print(type(female_read))

zeros = np.zeros((maxlength-female_read.shape[0]))
female_original = np.concatenate((female_read, zeros))
male_original = male_read
print(female_original.shape[0])
mix = female_original + male_original

mix_data = stft(mix, 512, 256, 512, window='hamming', center=False)
female_data = stft(female_original,512, 256, 512, window='hamming', center=False)
male_data = stft(male_original,512, 256, 512, window='hamming', center=False)
print (mix_data.shape)
print(female_data.shape)
print(male_data.shape)

action_matrix = (female_data)/(mix_data)
# error = female_data+male_data-mix_data
# print(error[:,1])
# print (action_matrix.shape)
# print(action_matrix[:,1])

action_matrix = action_matrix.T
mix_data = np.real(mix_data.T)
print('the shape of action_matrix', action_matrix.shape)

np.savetxt('target_action_matrix.ods', action_matrix, delimiter=',')

# separation = action_matrix*mix_data
# wavout_signal = istft(separation,  256, 512, window='hamming',center=False)
#
# wav_truth_signal = (female_original).tolist()
# wav_truth_noise = (male_original).tolist()
# wav_pred_signal = (wavout_signal).tolist()
# wav_pred_noise = (mix[0:wavout_signal.shape[0]] - wavout_signal).tolist()
# wav_mix = (mix).tolist()
#
# wav_truth_signal = matlab.double(wav_truth_signal)
# wav_truth_noise = matlab.double(wav_truth_noise)
# wav_pred_signal = matlab.double(wav_pred_signal)
# wav_pred_noise = matlab.double(wav_pred_noise)
# wav_mix = matlab.double(wav_mix)
#
# params = eng.BSS_EVAL(wav_truth_signal, wav_truth_noise, wav_pred_signal, wav_pred_noise, wav_mix)
# print(params)

train_data = np.zeros((1445,3,257))
for i in range(mix_data.shape[0]):
    if i == 0:
        s0 = mix_data[0, :]
        s1 = np.vstack((s0, s0))
        train_data[i] = np.vstack((s1, s0))
    elif i == mix_data.shape[0] - 1:
        new_state = np.concatenate((mix_data[i], mix_data[i], mix_data[i]))
        new_state = new_state.reshape(3, (mix_data[1].shape[0]))
        train_data[i] = new_state
    else:
        new_state = np.concatenate((mix_data[i-1], mix_data[i], mix_data[i+1]))
        new_state = new_state.reshape(3, (mix_data[1].shape[0]))
        train_data[i] = new_state

print (train_data.shape)

action_size=257
actor = create_actor_network(771, 257)

actor.fit(train_data, action_matrix, nb_epoch=50,batch_size=32)
print(actor.history)

print('now we save model')
actor.save_weights('pre_actormodel.h5', overwrite=True)
with open('pre_actormodel.json', 'w') as outfile:
    json.dump(actor.to_json(), outfile)


