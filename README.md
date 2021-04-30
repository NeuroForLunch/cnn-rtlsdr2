# cnn-rtlsdr2

Convolutional Neural Networks with TensorFlow 2 and Keras for signal classification using a RTL-SDR dongle.

Current pre-trained model is able to classify 4 kinds of signals: WFM, TV, DMR,  and "Others".

### TEST WITH PRETRAINED MODEL
> brew install python

```bash
› python -m venv --system-site-packages ./venv

› source ~/venv/bin/activate

› pip install --upgrade pip

› pip install -r requirements.txt

› python predict_scan.py
```

to scan entire band and predict signal types , or the full version scan:

```
python predict_scan.py --start 25000000 --stop 1750000000 --step 50000 --gain 20 --ppm 20 --threshold 0.9955
```

Some help is also available:
```
python predict_scan.py --help
```

### TRAIN YOUR OWN DATA

To train your own model, edit the settings in file [prepare_data.py] to set own frequencies of local stations and ppm error.
```
sdr.err_ppm = 20     # change it to yours

collect_samples(104000000, "wfm")
collect_samples(942200000, "gsm")
```

Then to obtain some samples run:
```
python prepare_data.py
```

Delete unnecessary folders under [/testing_data] and [/training_data] as they are responsible for classificator.
E.g., if you want to train only WFM and OTHER classes, delete everything, except:
- /training_data/wfm/
- /training_data/other/
- /testing_data/wfm/
- /testing_data/other/

Cleanup previous model checkpoint before starting a new training session (otherwise it will continue training old model).
```bash
echo y | del checkpoint
echo y | del rtlsdr-model.data-00000-of-00001
echo y | del rtlsdr-model.index
echo y | del rtlsdr-model.meta
```

Finally, we may now run training:
```
python train.py
```

A wise choice is to stop the training [ctrl+c], when validation loss becomes 0.1 - 0.01 or below. Lowest values shows better performance.
You may terminate training even after a few (20-30) epochs with values about 0.4 - 0.3 and evaluate the model.

It is better to obtain different samples of signals at different frequencies and gain levels. Edit [prepare_data.py] and run it again.
Then train the classifier again to see the difference. Feel free to sample your own signal classes to train a bigger model.



### KERAS VERSION

This is an optimized version of the network that reaches 99% accuracy while training.

```
python prepare_data.py
python train_keras.py
```




### Project evolution

First version of this project was built using adaptation of image classification network, as the RF signal is representating also in 2D .
I fed network with raw IQ samples, formed in a square as image, and even this gave me the model, doing it's job! This CNN graph was:
```
Conv2D (32*3*3) -> Conv2D (32*3*3) -> Conv2D (64*3*3) -> Dense (128) -> Dense (output)
```

Inspired by success, I began to try different preprocessing methods before feeding the network with complexity. Neural networks generally has
no idea, which input they serves, so I tried to form into image shape the following:

FFT data
```
iq_samples = np.fft.fft(iq_samples)
```

AM demodulation data
```
iq_samples = np.sqrt(np.real(iq_samples) ** 2 + np.imag(iq_samples) ** 2)
```

FM demodulation data
```
iq_samples = np.unwrap(np.angle(iq_samples))
iq_samples = np.diff(iq_samples)
```

and all of those gave me some results. FFT version converged very fast, in a 3-5 epochs, while AM demod version showed worst performance. Finally,
I've started googling to get more info and found the paper https://arxiv.org/pdf/1602.04105.pdf with all the CNN math and great explanations. Then
I adapted the network to match the paper one, and the graph now becomes:
```
Conv2D (64*1*3) -> Conv2D (16*2*3) -> Dense (128) -> Dense (output)
```

Feeding it with 1/4 sec raw IQ samples, sampled at 2.4 MSPS, and then decimated to a constant value of 48, left 12500 Hz bandwidth for classification.
