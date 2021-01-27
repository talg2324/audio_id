import pandas as pd 
import numpy as np 
import librosa
import librosa.display
import matplotlib.pyplot as plt

class Pre_processing() :

    def __init__(self, sr, n_mel_coeff):

        self.sr = int(sr) 
        self.frame_length = 2048
        self.hop_length = 512
        self.n_fft = 2048 
        self.window = 'hann' 
        self.n_mel_coeff = int(n_mel_coeff)
        
    def stats(self,x) :

        fetures = [
            'spectral_centroid',
            'spectral_bandwidth',
            'rms',
            'zero_crossing_rate'
            ]
        spectral_centroid = librosa.feature.spectral_centroid(x,sr = self.sr ,n_fft =self.n_fft, hop_length = self.hop_length ,window=self.window, center=True )
        spectral_centroid = spectral_centroid.flatten()

        spectral_bandwith = librosa.feature.spectral_bandwidth(x,sr = self.sr ,n_fft =self.n_fft, hop_length = self.hop_length ,window=self.window, center=True , p = 2)
        spectral_bandwith = spectral_bandwith.flatten()

        rms = librosa.feature.rms(x, frame_length = self.frame_length, hop_length = self.hop_length, center = True )
        rms = rms.flatten()

        zero_crossing_rate = librosa.feature.zero_crossing_rate(x ,frame_length = self.frame_length ,hop_length=self.hop_length ,center=True)
        zero_crossing_rate =zero_crossing_rate.flatten()

        stats_array = librosa.util.stack([spectral_centroid ,spectral_bandwith, rms, zero_crossing_rate], axis = 0)

        return fetures , stats_array

    def stft(self,x) :

        return librosa.stft(x, n_fft = self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window=self.window, center=True)

    def istft(self,x) :

        return librosa.istft(x, hop_length=self.hop_length, win_length=self.n_fft, window=self.window, center=True)
    
    def mel_coeff(self,x,num_of_derivatives) :

        _mel_coeff = librosa.feature.mfcc(x, sr= self.sr, n_mfcc= self.n_mel_coeff, dct_type=2, norm='ortho')

        if num_of_derivatives > 0 :

            all_mel_coeff = np.zeros([1+num_of_derivatives,_mel_coeff.shape[0],_mel_coeff.shape[-1]])
            all_mel_coeff[0,:] =_mel_coeff

            for i in range(1,1+num_of_derivatives) :

                all_mel_coeff[i,:,:] = librosa.feature.delta(_mel_coeff, order =i ,mode = 'interp')
    
            return all_mel_coeff
        else:

            return _mel_coeff

    def mel_spect(self,x, power_2_db = False) :

        if power_2_db :
            mel_spec =  librosa.feature.melspectrogram(x, sr=self.sr ,n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window=self.window, center=True, pad_mode='reflect', power=2.0)
            mel_spec = librosa.power_to_db(mel_spec, ref =np.max)

            return mel_spec

        else :
            return librosa.feature.melspectrogram(x, sr=self.sr ,n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window=self.window, center=True, pad_mode='reflect', power=2.0)
    
    def mel_spec_deriv(self, x) :

        return librosa.feature.delta(x, order = 1)

    def show_time_plot(self,x) :

        _ax = plt.gca()
        librosa.display.waveplot(x, sr=self.sr ,x_axis='time', ax= _ax)
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
    
    def my_dict(self,data) :
        mel = self.mel_spect(data, power_2_db=True)
        mel_dev_1 = self.mel_spec_deriv(mel)
        mel_dev_2 =self.mel_spec_deriv(mel_dev_1)
        feats, stats = self.stats(data)
        mfcc = self.mel_coeff(data,2)
        mfcc_0 = mfcc[0,:,:]
        mfcc_1 = mfcc[1,:,:]
        mfcc_2 = mfcc[2,:,:]
        features_dict = {
            'statistic_features' : stats,
            'statistic_feature_names' : feats,
            'stft' : np.log(np.abs(self.stft(data))),
            'mfcc' : mfcc_0,
            'del-mfcc' : mfcc_1,
            'del-del-mfcc' : mfcc_2,
            'mel_spec' : mel ,
            'mel_spec_dev_1' : mel_dev_1 ,
            'mel_spec_dev_2' : mel_dev_2 
        }

        return features_dict
    
def demo():

    data_file = './data/arbel1.csv'
    signal = pd.read_csv(data_file)
    _sr = 8000
    _mel_coeff = 10
    audio_model = Pre_processing(_sr,_mel_coeff)
            
    a = audio_model.my_dict(signal)
    audio_model.show_spec(a.get('mel_spec_dev_1') ,_x_axis = 'time', scale = 'log', spec_name = 'yariv')

#if __name__ == '__main__' :
    
    #demo()
    
    

