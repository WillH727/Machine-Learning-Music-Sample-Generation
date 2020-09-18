# Machine-Learning-Music-Sample-Generation

The goal of this project is to build a machine learning system that reads hundreds of samples and use this data to then generate its own samples based off the trainned data.

To change audio data into machine learn-able features, the files were decoded into raw amplitude arrays and then converted to Mel-frequency cepstral coefficients, which are a fourier transformation of a small time interval and logrymthic as that is how we heard sound.

# Long short-term memory

The first attempt was to build LSTM (Long short-term memory), which is a great model for finding relationships based off time incrementing data.
It works by LSTM neurons intreating with a pervious set of data based off the last incrememnt and is able to have long term memory by having 'roads' of little to no differiantion to stop gradient run off.

A simple LSTM was created consisting of a dense input layer, LSTM layor, relu layor and singular output layer. With dropout occuring between the LSTM and relu layor.

200 samples of drum loops were given, e.g.

and the result was this:

Clearly there is too much complexcity occuring for the LSTM to pick out on any patterns or alot more data is required.

# Generative adversarial network

Now having a look at the mel data,

its clear that the fed data looks more like an image to maybe a GAN (generative adversarial network) which be a much more natural fit.
A GAN works by having 2 neuro netoworks at feed back off each other, one tries to generate convolutions and build and image; the other tries to tell the real image from the fake image produce and each is back probagated based off this result.
