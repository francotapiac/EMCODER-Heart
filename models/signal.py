from turtle import window_height
import biosppy
import numpy as np
import pyhrv.tools as tools
from opensignalsreader import OpenSignalsReader
from pyhrv.hrv import hrv
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd



class Signal:
    
    def __init__(self, path, sampling):
        self.points_signal = self.get_signal(path)
        self.t,self.filtered_signal,self.rpeaks = self.get_rpeaks(self.points_signal,sampling)
        self.general_hrv = self.get_general_hrv(self.rpeaks,self.t,sampling)
        self.time_params = self.get_time_params(self.rpeaks, self.t)
        self.freq_params = self.get_freq_params(self.rpeaks, self.t)
        self.cardiac_coherence = self.get_cardiac_coherence(self.freq_params, self.rpeaks, self.t)
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
        t, filtered_signal, rpeaks = biosppy.signals.ecg.ecg(points_signal, sampling_rate=sampling, show= False)[:3]
        return t,filtered_signal,rpeaks

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

    def get_cardiac_coherence(self, freq_params, rpeaks, t):
        
        # Crea un rango auxiliar de frecuencias según la documentación
        # La coherencia cardiaca se evalua en el rango 0.04 al 0.26
        fbands_aux = {'ulf': (0.0, 0.01),'vlf': (0.01, 0.04), 'lf': (0.04, 0.26), 'hf': (0.26, 0.4)}
        aux_freq_params = fd.frequency_domain(rpeaks=t[rpeaks], show= False,fbands=fbands_aux)
        
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
        coherence_freq_params = fd.frequency_domain(rpeaks=t[rpeaks], show= False,fbands=fbands_coherence)
        
        # Se implementa la ecuación: Ratio coherence = Peak power/ (Total power - Peak power)
        peak_power = coherence_freq_params["fft_abs"][2] # Se obtiene el peak power del lf (rango evaluado)
        total_power = freq_params["fft_total"] # Potencia de toda la señal
        coherence_ratio = peak_power/(total_power - peak_power)
        return coherence_ratio
        
    def show_heart_coherence(self, coherence_ratio):
            if(coherence_ratio <= 0.5):
                return 1,"Incoherencia"
            