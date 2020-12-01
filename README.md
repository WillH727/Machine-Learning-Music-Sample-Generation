# Machine-Learning-Music-Sample-Generation

The goal of this project is to build a machine learning system that reads hundreds of samples and use this data to then generate its own samples based off the trained data. The focus is on making weird low bit sounds that resembles general amplitude and pitch envelopes, as I don't have the computing power or data for higher quality sounds, but this should be fun anyway.

To change audio data into machine learn-able features, the files were decoded into raw amplitude arrays and then converted to Mel-frequency cepstral coefficients, which are a Fourier transformation of a small time interval and logarithmic as that is how we heard sound.

# Long short-term memory, GAN

The first attempt was to build LSTM (Long short-term memory), which is a great model for finding relationships based off time incrementing data.
It works by LSTM neurons intreating with a pervious set of data based off the last increment and can have long term memory by having 'roads' of little to no differential to stop gradient run off.(https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
But just an LSTM couldn’t learn enough from the small number of complex data I have (about 200 samples), so I changed learning method to generative adversarial network (GAN). This has the benefit of learning multiply things off the same data, but obviously creates overfitting problems, but for this project I don't care about overfitting if it creates cool sounds.
A GAN works by having too networks, one that takes noise and tries to generate data and another network that is given the real data and output of the first network and tries to figure out which is real. So I built both networks with a LSTM layer with the same number of nodes as Mel-frequency cepstral coefficients with no output layer for generator and one for discriminator.

200 samples of drum loops were given, e.g.

[elephant](examples/last.wav)

and the result was this:

Clearly there is too much complexcity occuring for the LSTM to pick out on any patterns or alot more data is required.

# Generative adversarial network

Now having a look at the mel data,

its clear that the fed data looks more like an image to maybe a GAN (generative adversarial network) which be a much more natural fit.
A GAN works by having 2 neuro netoworks at feed back off each other, one tries to generate convolutions and build and image; the other tries to tell the real image from the fake image produce and each is back probagated based off this result.
