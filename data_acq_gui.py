import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import write
import time
import pandas as pd
import sounddevice as sd
from scipy.io.wavfile import write
import tkinter as tk

class Recorder():
        def __init__(self):

            # Init params
            start_name = ''
            start_channel = 1
            start_fs = 8000    #Hz

            self.fs = start_fs
            self.channel = start_channel
            self.name = start_name

            # UI
            self.screen = tk.Tk() 
            self.screen.geometry('{}x{}'.format(800,300))
            self.screen.title('Data Acquisition App' )
            tk.Label(self.screen, text = 'Recording Device' ).grid(row = 1)

            tk.Label(self.screen, text = 'Recording Time (seconds) : ' ).grid( row = 1,column = 0)
            tk.Label(self.screen, text = 'Sampling (Hz) :').grid(row = 1, column = 2)
            tk.Label(self.screen, text = 'Mic channel :').grid(row = 1, column = 4)
            tk.Label(self.screen, text = 'Name :').grid(row = 2)

            entry1 = tk.Entry(width = 10)
            entry1.insert(0,str(self.fs))
            entry1.grid( row = 1, column = 3)

            entry2 = tk.Entry(width = 10)
            entry2.insert(0,str(self.channel))
            entry2.grid( row = 1, column = 5)

            self.entry3 = tk.Entry(width = 15)
            self.entry3.insert(0,start_name)
            self.entry3.grid( row = 2, column = 1)

            self.record_stream = []
            self.complete = False

            start_rec_button = tk.Button(self.screen, text="Start recording",width=10, command=self.start_record)
            start_rec_button.grid( row = 3 , column = 1)


            end_rec_button = tk.Button(self.screen, text="End recording",width=10, command=self.end_record)
            end_rec_button.grid( row = 3 , column = 2)


            quit_button = tk.Button(self.screen, text="Quit",width=10, command= self.finish_session)
            quit_button.grid( row = 3 , column = 3)

        def record_block(self, indata, frames, time, status):

            self.record_stream.append(indata.copy())

        def start_record(self):
            
            self.rec = sd.InputStream(samplerate=self.fs, channels=self.channel, blocksize=self.fs, callback=self.record_block)
            self.rec.start()

        def end_record(self):
            
            self.rec.stop()

            self.record_stream = np.array(self.record_stream).flatten()

            self.complete = True

        def finish_session(self):

            self.rec.abort()

            self.name = self.entry3.get()

            if self.complete:

                np.savetxt('./' + self.name +'.csv', self.record_stream, delimiter = ',')   
                write('./' + self.name +'.wav', self.fs, self.record_stream)    

            self.screen.quit()

        def run(self):

            self.screen.mainloop()
            
if __name__ == '__main__':
    
    ui = Recorder()

    ui.run()
