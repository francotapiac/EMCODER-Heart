from queue import Full
from turtle import window_height
from typing import List
import biosppy
import numpy as np
import pyhrv
from pyhrv.hrv import hrv
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import math
import heartpy as hp



class Signal:
    
    def __init__(self, path, sampling):
        self.points_signal = self.get_signal(path)
        self.t,self.filtered_signal,self.rpeaks = self.get_rpeaks(self.points_signal,sampling)
        self.nni = self.get_nni(self.rpeaks, self.t)
        self.general_hrv = self.get_general_hrv(self.rpeaks,self.t,sampling)
        
        # Segmentando señal y tiempo
        #self.segmented_signal = self.segment_signal(self.filtered_signal, sampling, 0.5)
        #self.segmented_time  = self.segment_time(self.segmented_signal, self.t)
        self.segment_nni = self.segment_nni(self.nni, 9)

        #self.process_signal(self.segment_nni )
       
        self.time_params = self.get_time_params(self.rpeaks, self.t)
        self.freq_params = self.get_freq_params(self.rpeaks, self.t)
        self.cardiac_coherence = self.get_cardiac_coherence(self.nni)
        
        self.hrdata = hp.get_data(path)
        self.working_data, self.measures = hp.process(self.hrdata, 1000.0)
        hp.plotter(self.working_data, self.measures)
        self.newworking_data, self.newmeasures = hp.process_segmentwise(self.hrdata, sample_rate=1000.0, segment_width = 14, segment_overlap = 0.25, calc_freq=True, reject_segmentwise=True, report_time=True)
        
        pass
    
    """
    Obtiene la señal cardiaca de un archivo
    @param path:    archivo con señal cardiaca
    @return:        arreglo con puntos de la señal
    """      
    def get_signal(self,path):
        points_signal = []   
        with open(path, 'r') as f:
            for line in f:
                points_signal.append(float(line.strip()))
        return points_signal
    
    
    """
    Obtiene series de R-Peaks usando biosppy
    @param signal:  señal cardiaca
    @param sampling:frecuencia de muestreo
    @return:        rango de tiempo de la señal, señal filtrada y rpeaks  
    """                
    def get_rpeaks(self,points_signal, sampling):
        print(len(points_signal))
        t, filtered_signal, rpeaks = biosppy.signals.ecg.ecg(points_signal, sampling_rate=sampling, show= False)[:3]
        return t,filtered_signal,rpeaks
    
    
    def get_nni(self, rpeaks, t):
        nni = pyhrv.tools.nn_intervals(t[rpeaks])
        return nni
    
    def segment_nni(self, nni, duration):
        segments, control = pyhrv.utils.segmentation(nni=nni, duration=duration, full=True)    
        # Print control variable
        print("Segmentation?", control)

        # Print segments
        for i, segment in enumerate(segments):
            print("Segment %i" % i)
            print(segment)
        return segments
    
    def segment_time(self, segmented_signal, t):
        count_t = 0
        t_segment = []
        for i in range(0, len(segmented_signal)):
            # Segmentando el tiempo en rangos iguales a los segmentos de la señal
            t_segment.append(t[count_t: len(segmented_signal[i]) + count_t ])
            count_t = len(segmented_signal[i]) + count_t
        return t_segment
        
    
        
    """
    Segmenta la señal en un rango de tiempo definido
    @param points_signal:   Puntos de la señal
    @param sampling:        Frecuencia de muestreo
    Qparam window:          Ventana que se desplaza a lo largo de la señal
    @return:        Señal segmentada según la ventana de tiempo
    """
    def segment_signal(self, points_signal , sampling, window):
        shiftLen=window
        duration=int(window*sampling)
        dataOverlap = (window-shiftLen)*sampling
        numberOfSegments = int(math.ceil((len(points_signal)-dataOverlap)/(duration-dataOverlap)))
        print(numberOfSegments)
        #print(data.shape)
        tempBuf = [points_signal[i:i+duration] for i in range(0,len(points_signal),(duration-int(dataOverlap)))]
        tempBuf[numberOfSegments-1] = np.pad(tempBuf[numberOfSegments-1],(0,duration-tempBuf[numberOfSegments-1].shape[0]),'constant')
        tempBuf2 = np.vstack(tempBuf[0:numberOfSegments])
        
        return tempBuf2
    
    
    def process_signal(self, segmented_signal):
        for i in range(0, len(segmented_signal)):
            cardiac_coherence_segment = self.get_cardiac_coherence(segmented_signal[i])
        return 1
        
        

    """
    Calcula todos los parametros del HRV
    @param rpeaks:  tiempos R-Peaks [ms]
    @param t:       tiempo de muestreo de la señal
    @param sampling:frecuencia de muestreo de la señal
    @return:        todos los parametros del hrv
    """         
    def get_general_hrv(self,rpeaks, t, sampling):
        rpeaks_results = hrv(rpeaks=t[rpeaks], sampling_rate=sampling, show=False)
        return rpeaks_results
    
    """
    Calcula los parametros en el tiempo de la señal
    @param rpeaks:  tiempos R-Peaks [ms]
    @param t:       tiempo de muestreo de la señal
    @return:        parametros en el tiempo de la señal
    """      
    def get_time_params(self, rpeaks, t):
        time_params = td.time_domain(rpeaks=t[rpeaks])
        return time_params
        
    """
    Calcula los parametros en la frecuencia de la señal
    @param rpeaks:  tiempos R-Peaks [ms]
    @param t:       tiempo de muestreo de la señal
    @return:        parametros en la frecuencia de la señal
    """   
    def get_freq_params(self, rpeaks, t):
        freq_params = fd.welch_psd(rpeaks=t[rpeaks], show= False)
        return freq_params

    """
    Obtiene la coherencia cardiaca a partirar de los tiempos R-Peaks [ms]
    @param rpeaks:  tiempos R-Peaks [ms]
    @param t:       tiempo de muestreo de la señal
    @return:        valor de la coherencia cardiaca 
    """   
    def get_cardiac_coherence(self, nni):
        
        # Crea un rango auxiliar de frecuencias según la documentación
        # La coherencia cardiaca se evalua en el rango 0.04 al 0.26
        fbands_aux = {'ulf': (0.0, 0.01),'vlf': (0.01, 0.04), 'lf': (0.04, 0.26), 'hf': (0.26, 0.4)}
        aux_freq_params = fd.frequency_domain(nni = nni, show= False,fbands=fbands_aux)
        
        # Se obtiene el peak (punto de mayor potencia) presente en los 0.04 al 0.26 hz
        peak_aux = aux_freq_params["fft_peak"][2]
        upper_window = 0
        low_window = 0
        
        # Se establece la ventana de 0.03 hz.  (0.015 bajo el peak y 0.015 sobre el peak)
        # Si la ventana sobrepasa el rango más alto, la ventana se ajusta para que llegue al peak alto
        if(peak_aux >= 0.245):
            upper_window = 0.26
            low_window = peak_aux - 0.015
        # Si la ventana sobrepasa el rango más bajo, la ventana se ajusta para que llegue al peak bajo
        elif(peak_aux <= 0.055):
            upper_window = peak_aux + 0.015
            low_window = 0.04
        else:
            upper_window = peak_aux - 0.015
            low_window = peak_aux + 0.015
        
        # Se obtiene la potencia correspondiente al peak evaluado sobre la ventana (lf en este caso, debido a que la función no permite cambiar el nombre)
        fbands_coherence = {'ulf': (0.0, 0.01),'vlf': (0.01, low_window), 'lf': (low_window, upper_window), 'hf': (upper_window, 0.4)}
        coherence_freq_params = fd.welch_psd(nni = nni, show= False,fbands=fbands_coherence)
        
        # Se implementa la ecuación: Ratio coherence = Peak power/ (Total power - Peak power)
        peak_power = coherence_freq_params["fft_abs"][2] # Se obtiene el peak power del lf (rango evaluado)
        total_power = coherence_freq_params["fft_total"] # Potencia de toda la señal
        coherence_ratio = peak_power/(total_power - peak_power)
        print(coherence_ratio)
        return coherence_ratio
    
   
        
    def show_heart_coherence(self, coherence_ratio):
            if(coherence_ratio <= 0.5):
                return 1,"Incoherencia"
            