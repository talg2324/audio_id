import pandas as pd 
import numpy as np 
import librosa
import librosa.display
import matplotlib.pyplot as plt

class Pre_processing() :

    def __init__(self,data, sr, n_mel_coeff, frame_length = 2048 ,hop_length = 512, n_fft = 2048 , window = 'hann' ):

        self.raw_data  = data
        self.data = self.raw_data.to_numpy().T
        self.data = self.data.flatten()

        self.signal_length = len(self.data)
        self.sr = int(sr) 
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.window = window
        self.n_mel_coeff = int(n_mel_coeff)
        
    def stats(self) :

        fetures = [
            'spectral_centroid',
            'spectral_bandwidth',
            'rms',
            'zero_crossing_rate'
            ]
        spectral_centroid = librosa.feature.spectral_centroid(self.data,sr = self.sr ,n_fft =self.n_fft, hop_length = self.hop_length ,window=self.window, center=True )
        spectral_centroid = spectral_centroid.flatten()

        spectral_bandwith = librosa.feature.spectral_bandwidth(self.data,sr = self.sr ,n_fft =self.n_fft, hop_length = self.hop_length ,window=self.window, center=True , p = 2)
        spectral_bandwith = spectral_bandwith.flatten()

        rms = librosa.feature.rms(self.data, frame_length = self.frame_length, hop_length = self.hop_length, center = True )
        rms = rms.flatten()

        zero_crossing_rate = librosa.feature.zero_crossing_rate(self.data ,frame_length = self.frame_length ,hop_length=self.hop_length ,center=True)
        zero_crossing_rate =zero_crossing_rate.flatten()

        stats_array = librosa.util.stack([spectral_centroid ,spectral_bandwith, rms, zero_crossing_rate], axis = 0)

        return fetures , stats_array

    def stft(self) :

        return librosa.stft(self.data, n_fft = self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window=self.window, center=True)

    def istft(self,x) :

        return librosa.istft(x, hop_length=self.hop_length, win_length=self.n_fft, window=self.window, center=True)
    
    def mel_coeff(self,num_of_derivatives) :

        _mel_coeff = librosa.feature.mfcc(self.data, sr= self.sr, n_mfcc= self.n_mel_coeff, dct_type=2, norm='ortho')

        if num_of_derivatives > 0 :

            all_mel_coeff = np.zeros([1+num_of_derivatives,_mel_coeff.shape[0],_mel_coeff.shape[-1]])
            all_mel_coeff[0,:] =_mel_coeff

            for i in range(num_of_derivatives) :

                all_mel_coeff[i,:,:] = librosa.feature.delta(_mel_coeff ,width = _mel_coeff.shape[-1],order =i+1 ,mode = 'interp')
    
            return all_mel_coeff
        else:

            return _mel_coeff

    def mel_spect(self,power_2_db = False) :

        if power_2_db :
            
            mel_spec =  librosa.feature.melspectrogram(self.data, sr=self.sr ,n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window=self.window, center=True, pad_mode='reflect', power=2.0)
            mel_spec = librosa.power_to_db(mel_spec, ref =np.max)

            return mel_spec

        else :

            return  librosa.feature.melspectrogram(self.data, sr=self.sr ,n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window=self.window, center=True, pad_mode='reflect', power=2.0)
    
    def mel_spec_deriv(self, mel) :

        return librosa.feature.delta(mel, width = mel.shape[-1] ,order = 1,mode = 'interp')

    def show_time_plot(self) :

        _ax = plt.gca()
        librosa.display.waveplot(self.data, sr=self.sr ,x_axis='time', ax= _ax)
        _ax.set(title = 'Audio signal')

        plt.show()

        return 

    def show_spec(self,x,_x_axis = None ,scale = None, spec_name = '') :

        fig = plt.figure()
        _ax = plt.gca()
        img = librosa.display.specshow(x, x_axis=_x_axis, y_axis=scale , sr=self.sr, hop_length=self.hop_length, fmin=None, fmax=None, bins_per_octave=12, key='C:maj', ax=_ax)
        _ax.set(title = spec_name+' Spectogram')

        fig.colorbar(img, ax=_ax, format="%+2.f dB")
        plt.show()

        return
    

         

if __name__ == '__main__' :

    data_file = './data/yaya.csv'
    signal = pd.read_csv(data_file)
    audio_model = Pre_processing(signal,8000,10)
            
    a = audio_model.mel_spect(power_2_db = True)
    audio_model.show_spec(a ,_x_axis = 'time', scale = 'log', spec_name = 'yariv')
    
    print(1)
    
    

