import librosa, librosa.display, librosa.feature
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import os
from tqdm import tqdm
import random

"""
--Get data set
"""
batch_size = 5

# hop length is 512 for default
hop_length = 512
#default is 2048
n_fft = 2048
n_mfcc = 26

mypath = 'C:\\Users\\willi\\Documents\\Python_Scripts\\LSTM_gen\\sample_data'

#lists file names in mypath dir
file_names = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

#coverts wav's to  list of np arrays
wf_data = []

for name in file_names:
	signal, sample_rate = librosa.load(mypath+'\\'+'DNB_BREAK_01.wav')
	
	samp = librosa.feature.mfcc(signal,
							 sr=sample_rate,
							 n_fft=n_fft,
							 n_mfcc=n_mfcc,
							 hop_length=hop_length)
	samp = np.transpose(samp)
	wf_data.append(samp)

last_samp_size = len(samp)

samp = np.transpose(samp)
librosa.display.specshow(samp,sr=sample_rate,hop_length=hop_length)
plt.show()

samp = librosa.feature.inverse.mfcc_to_audio(samp,
											 sr=sample_rate,
											 n_fft=n_fft)

librosa.output.write_wav('last.wav', samp, sample_rate)

data_dim = wf_data[0].shape
print("Samples Loaded")

def split_arr(arr, sequence_len):
	no_samples = len(arr)//sequence_len
	
	if len(arr) % no_samples != 0:
		arr = arr[0:no_samples*sequence_len]
	
	arr = np.split(arr,no_samples,axis=0)
	
	x = np.zeros((no_samples,sequence_len//2,n_mfcc))
	y = np.zeros((no_samples,sequence_len//2,n_mfcc))
	
	for i in range(no_samples):
		if len(arr[i])%2 != 0:
			arr[i] = np.delete(arr[i],-1,axis=0)
		x[i], y[i] = np.split(arr[i],2,axis=0)
	
	return x,y

def normilize_data(data):
	sample_len = min(len(i[:]) for i in data)
		
	for i in range(len(data)):
		data[i] = data[i][0:sample_len]
	
	return data

def get_samples(data, batch_size, sequence_len):
	data = normilize_data(data)
	
	random.shuffle(data)
	
	x = [split_arr(sample, sequence_len)[0] for sample in data]
	y = [split_arr(sample, sequence_len)[1] for sample in data]
	#(no_sets_of_seq,no_of_seq, len_squ, n_mfcc)
	
	x = np.reshape(x,(len(x)*len(x[0]),sequence_len//2,n_mfcc))
	y = np.reshape(y,(len(y)*len(y[0]),sequence_len//2,n_mfcc))
	#(no_sets_of_seq*no_of_seq, len_squ, n_mfcc)
	
	x_batches = []
	y_batches = []
	
	for i in range(len(x)//batch_size):
		x_batches.append(tf.convert_to_tensor(
			[x[batch_size*i+j] for j in range(batch_size)]))
		y_batches.append(tf.convert_to_tensor(
			[y[batch_size*i+j] for j in range(batch_size)]))
	
	return x_batches, y_batches


"""
--Make model
"""
def build_model(rnn_units,batch_size,data_dim):
	model = tf.keras.Sequential([
		#input layer
		tf.keras.layers.Dense(rnn_units,
							batch_input_shape=[batch_size, None, data_dim[1]]),
		
		#LSTM layer
		tf.keras.layers.LSTM(
			rnn_units, 
			return_sequences=True, 
			recurrent_initializer='glorot_uniform',
			recurrent_activation='sigmoid',
			stateful=True,
			),
		tf.keras.layers.Dropout(0.2),
		
		tf.keras.layers.Dense(rnn_units, activation='relu'),
		
		#output layer
		tf.keras.layers.Dense(data_dim[1])
		])
	return model
	

"""
--Train model
"""
sequence_len = 20

x_data, y_data = get_samples(wf_data, batch_size, sequence_len)


# Optimization parameters:
num_training_iterations = len(x_data)
learning_rate = 1e-2

rnn_units = 4096

# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

model = build_model(rnn_units,batch_size,data_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate)

@tf.function
def train_step(x, y): 
	# Use tf.GradientTape()
	with tf.GradientTape() as tape:
		
		y_hat = model(x)
		
		loss = tf.keras.losses.mse(y,y_hat)
		
	grads = tape.gradient(loss, model.trainable_variables)
	
	# Apply the gradients to the optimizer so it can update the model accordingly
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
	return loss

history = []
i = 0

for iter in tqdm(range(num_training_iterations)):	
	loss = train_step(x_data[i], y_data[i])
	history.append(loss.numpy().mean())
	i += 1
	if iter % 100 == 0:
		model.save_weights(checkpoint_prefix)

model.save_weights(checkpoint_prefix)

plt.figure()
plt.plot(history)
plt.show()

"""
--Generate sounds
"""

model = build_model(rnn_units,1,data_dim)

# Restore the model weights for the last checkpoint after training
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None, data_dim[1]]))

def generate_sample(model, starting_value,generation_length,data_dim):
	
	input_eval = tf.expand_dims(starting_value,0)
	generated_wf = np.zeros((generation_length,data_dim[1]))
	
	model.reset_states()
	tqdm._instances.clear()
	
	for i in tqdm(range(generation_length)):
		
		prediction = model(input_eval)
		
		prediction = tf.squeeze(prediction,0)
		
		generated_wf[i] = prediction
		
		input_eval = tf.expand_dims(prediction, 0)
	
	return generated_wf

start = 50*np.random.rand(1,data_dim[1])
#start = np.zeros((1,data_dim[1]))

gen = generate_sample(model,start, last_samp_size, data_dim)

check = gen

gen = np.transpose(gen)

plt.figure()
librosa.display.specshow(gen,sr=sample_rate,hop_length=hop_length)
plt.show()

gen = librosa.feature.inverse.mfcc_to_audio(gen,
											 sr=sample_rate,
											 n_fft=n_fft)

librosa.output.write_wav('gen.wav', gen, sample_rate)

