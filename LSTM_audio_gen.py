import librosa, librosa.display, librosa.feature
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import os
import time

"""
--Get data set
"""
# hop length is 512 for default
hop_length = 64
#default is 2048
n_fft = 4096
n_mfcc = 52

mypath = 'C:\\Users\\willi\\Documents\\Python_Scripts\\LSTM_gen\\sample_data'

#lists file names in mypath dir
file_names = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

#coverts wav's to  list of np arrays
wf_data = []

for name in file_names:
	signal, sample_rate = librosa.load(mypath+'\\'+name)
	
	samp = librosa.feature.mfcc(signal,
							 sr=sample_rate,
							 n_fft=n_fft,
							 n_mfcc=n_mfcc,
							 hop_length=hop_length)
	samp = np.transpose(samp)
	wf_data.append(samp)

def normilize_data(data):
	norm_value = max([np.amax(np.abs(i)) for i in data])
	
	data = [i / norm_value for i in data]
	
	return data, norm_value

def denormilize_data(data, norm_value):
	data = data * norm_value
	
	return data

wf_data, norm_value = normilize_data(wf_data)

last_samp_size = len(samp)

samp = np.transpose(samp)

for i in range(0,len(wf_data),50):
	plt.figure()
	librosa.display.specshow(np.transpose(wf_data[i]),sr=sample_rate,hop_length=hop_length)
	plt.show()

samp = librosa.feature.inverse.mfcc_to_audio(samp,
											 sr=sample_rate,
											 n_fft=n_fft)

librosa.output.write_wav('last.wav', samp, sample_rate)

train_dataset = [tf.convert_to_tensor(i, dtype=tf.float32) for i in wf_data]

print("Samples Loaded")

"""
--Build models
"""
noise_feat = 26

def build_generator(rnn_units):
	model = tf.keras.Sequential([		
		tf.keras.layers.LSTM(
			n_mfcc, 
			return_sequences=True,
			stateful=True
			)
		])
	return model

def build_discriminator(rnn_units):
	model = tf.keras.Sequential([		
		tf.keras.layers.LSTM(
			n_mfcc, 
			return_sequences=True,
			stateful=True
			),
		
		tf.keras.layers.Dense(1)
		
		])
	return model

"""
--Train model
"""
rnn_units = 26
learning_rate = 1e-3

generator = build_generator(rnn_units)
discriminator = build_discriminator(rnn_units)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

noise_dim = [1, 20, noise_feat]
noise_dim_tf = tf.convert_to_tensor([1, 10, noise_feat],dtype=tf.int32)

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal(noise_dim)

@tf.function
def train_step(sample,sample_noise):
	
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	def discriminator_loss(real_output, fake_output):
	    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	    total_loss = real_loss + fake_loss
	    return total_loss

	def generator_loss(fake_output):
	    return cross_entropy(tf.ones_like(fake_output), fake_output)
	
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_samps = generator(sample_noise, training=True)
		real_output = discriminator(sample, training=True)
		fake_output = discriminator(generated_samps, training=True)

		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
	
	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
	
	return gen_loss, disc_loss
	
def generate_audio(model, test_input):
	# Notice `training` is set to False.
	# This is so all layers run in inference mode (batchnorm).
	predictions = model(test_input, training=False)
	
	audio_pred = np.transpose(predictions.numpy()[0])
	
	audio_pred_n = denormilize_data(audio_pred, norm_value)
	
	audio_raw = librosa.feature.inverse.mfcc_to_audio(audio_pred_n,
											 sr=sample_rate,
											 n_fft=n_fft)

	librosa.output.write_wav('generated.wav', audio_raw, sample_rate)
	
	plt.figure()
	librosa.display.specshow(audio_pred_n,sr=sample_rate,hop_length=hop_length)
	plt.show()


def train(dataset, epochs):
	
	gen_loss_arr = []
	discr_loss_arr = []
	
	for epoch in range(epochs):
		start = time.time()
		
		for train_samp in train_dataset:
			noise = tf.random.normal([1, train_samp.shape[0], noise_feat])
			train_samp = tf.expand_dims(train_samp, axis=0)
			gen_loss, discr_loss = train_step(train_samp, noise)
			gen_loss_arr.append(gen_loss)
			discr_loss_arr.append(discr_loss)
		
		# Save the model every 15 epochs
		if (epoch + 1) % 15 == 0:
			checkpoint.save(file_prefix = checkpoint_prefix)
  
		print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
	
	# Generate after the final epoch
	generate_audio(generator,seed)
	
	plt.figure()
	plt.subplot(211)
	plt.plot(gen_loss_arr)
	plt.xlabel('Gen Loss')
	plt.subplot(212)
	plt.plot(discr_loss_arr)
	plt.xlabel('Discr Loss')
	plt.show()

EPOCHS = 150
#get more data
train(train_dataset, EPOCHS)
