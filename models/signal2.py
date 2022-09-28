from pyhrv.hrv import hrv
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import biosppy
import math
import numpy as np
import pyhrv
import pandas as pd
 
import models.feature as  mf
import models.cardiacCoherence as mc


class Signal2:
    
    def __init__(self, name_signal, path, type_signal, sampling, number_segment):
        self.name_signal = 0
        self.points_signal = self.get_signal(path)
        self.name_signal = self.get_name_signal(type_signal)
        self.t,self.filtered_signal,self.rr_peaks = self.get_rpeaks(self.points_signal, type_signal, sampling)
        self.signal_segment, self.t_segment = self.create_windows_signal(self.filtered_signal,self.t,12,500)
        #self.signal_segment = self.segment_signal(self.filtered_signal,sampling,number_segment)
        #self.t_segment = self.segment_time(self.signal_segment,self.t)
     
        self.time_line = self.process_signal(self.signal_segment, self.t_segment,type_signal, sampling)
        self.PIns = []
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
    Define el nombre la señal según su tipo
    @param type_signal: tipo de señal
    @return:            string con nombre de señal
    """
    def get_name_signal(self, type_signal):
        if(type_signal == 1):
            return "ECG"
        else:
            return "PPG"
        
    """
    Obtiene series de R-Peaks usando biosppy
    @param signal:  señal cardiaca
    @param type_signal: tipo de señal (ECG y PPG)
    @param sampling:frecuencia de muestreo
    @return:        rango de tiempo de la señal, señal filtrada y rpeaks  
    """                
    def get_rpeaks(self,points_signal, type_signal, sampling):
        if(type_signal== 1):
            t, filtered_signal, rpeaks = biosppy.signals.ecg.ecg(points_signal, sampling_rate=sampling, show= False)[:3]
            return t,filtered_signal,rpeaks
        if(type_signal == 2):
            t, filtered_signal, rpeaks = biosppy.signals.ppg.ppg(points_signal, sampling_rate=sampling, show= False)[:3]
            return t,filtered_signal,rpeaks
    """
    Obtiene intervalos nni para el calculo de la coherencia cardíaca
    @param rpeaks:  valor de los peaks r
    @param t:       tiempo de la señal
    @return:        intervalos nni
    """
    def get_nni(self, rpeaks, t):
        nni = pyhrv.tools.nn_intervals(t[rpeaks])
        return nni    
    
    
    """
    Calcula los parametros en el tiempo de la señal
    @param rpeaks:  tiempos R-Peaks [ms]
    @param t:       tiempo de muestreo de la señal
    @return:        parametros en el tiempo de la señal
    """      
    def get_time_params(self, rpeaks, t):
        time_params = td.time_domain(rpeaks=t[rpeaks], show=False)
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
    Segmenta la señal en un rango de tiempo definido
    @param points_signal:   Puntos de la señal
    @param sampling:        Frecuencia de muestreo
    Qparam window:          Ventana que se desplaza a lo largo de la señal
    @return:        Señal segmentada según la ventana de tiempo
    """
    def segment_signal(self, points_signal , sampling, window):
        shiftLen=0.5
        duration=int(window*sampling)
        dataOverlap = (window-shiftLen)*sampling
        numberOfSegments = int(math.ceil((len(points_signal)-dataOverlap)/(duration-dataOverlap)))
        print(numberOfSegments)
        #print(data.shape)
        tempBuf = [points_signal[i:i+duration] for i in range(0,len(points_signal),(duration-int(dataOverlap)))]
        tempBuf[numberOfSegments-1] = np.pad(tempBuf[numberOfSegments-1],(0,duration-tempBuf[numberOfSegments-1].shape[0]),'constant')
        tempBuf2 = np.vstack(tempBuf[0:numberOfSegments])
        return tempBuf2
            
    """
    Segmenta el tiempo de acuerdo a los nuevos segmentos creados
    @param segmented_signal: señal segmentada según una cantidad de intervalos
    @param t:  arreglo de tiempo de la señal original
    @return: tiempo segmentado 
    """
    def segment_time(self, segmented_signal, t):
        count_t = 0
        t_segment = []
        for i in range(0, len(segmented_signal)):
            # Segmentando el tiempo en rangos iguales a los segmentos de la señal
            t_segment.append(t[count_t: len(segmented_signal[i]) + count_t ])
            count_t = len(segmented_signal[i]) + count_t
        return t_segment
    """
    Segmenta la señal según una ventana de tiempo. La ventana se mueve en pequeños segmentos
    de tiempo y se crea un nuevo arreglo para ser procesado
    @param signal: señal original
    @param t:  arreglo de tiempo de la señal original
    @param windo_size: ventana de tiempo para mover la señal
    @param shif: largo de tiempo que se mueve la ventana
    @return: señal segmentada con respectivo tiempo
    
    """
    def create_windows_signal(self, signal, t, window_size, shif):
        signal_segment = []
        t_segment = []
        init_pos = 0
        final_pos = window_size*1000
        while final_pos <= len(t):
            signal_segment.append(signal[int(init_pos):int(final_pos)])
            t_segment.append(t[int(init_pos):int(final_pos)])
            init_pos = init_pos + shif
            final_pos = final_pos + shif
        return signal_segment,t_segment
            
         
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
        return coherence_ratio
    
    def freqTimeToDataframe(self, time_ecg,freq_ecg):
        vlf = freq_ecg["fft_abs"][0]
        lf = freq_ecg["fft_abs"][1]
        hf = freq_ecg["fft_abs"][2]
        lf_hf = lf/hf
        fft_total = freq_ecg["fft_total"]
        hr_mean = time_ecg["hr_mean"]
        hr_min = time_ecg["hr_min"]
        hr_max = time_ecg["hr_max"]
        sdnn = time_ecg["sdnn"]
        rmssd = time_ecg["rmssd"]
        sdsd = time_ecg["sdsd"]
        pnn50 = time_ecg["pnn50"]
        
        freq_df=pd.DataFrame({'vlf': [vlf],
                                'lf': [lf],
                                'hf': [hf],
                                'lf/hf': [lf_hf/hf],
                                'fft_total': [fft_total]
                                })
        time_df=pd.DataFrame({'hr_mean': [hr_mean],
                                'hr_min': [hr_min],
                                'hr_max':[hr_max],
                                'sdnn':[sdnn],
                                'rmssd':[rmssd],
                                'sdsd':[sdsd],
                                'pnn50':[pnn50]

                                })
        return time_df,time_df
    """
    Procesa la señal calculando las Features, la coherencia cardiaca y las emociones por cada segmento
    @param signal_segment: señal segmentada
    @param t_segment: tiempo segmentado según señal
    @param type_signal: tipo de señal (1: ECG, 2:PPG)
    @param sampling: frecuencia de muestreo
    @return: línea de tiempo con señal procesada
    """
    def process_signal(self, signal_segment, t_segment, type_signal, sampling):
        time_line = []
        start_time = 0
        for segment, time in zip(signal_segment,t_segment):
            element_time_line= []
            end_time = time[-1]
            print(end_time)
            # Obteniendo rrpeaks y nni
            t,filtered_signal,rr_peaks = self.get_rpeaks(segment, type_signal, sampling)
            nni = self.get_nni(rr_peaks, t)            
           
            # Obteniendo atributos del dominio de la frecuencia y tiempo de la señal
            time_params = self.get_time_params(rr_peaks,t)
            freq_params = self.get_freq_params(rr_peaks,t)
            time_df, freq_df = self.freqTimeToDataframe(time_params,freq_params)
            # Calculando la coherencia cardíaca
            ratio_coherence = self.get_cardiac_coherence(nni)

            # Creando objetos Feature y CardiacCoherence
            feature = mf.Feature(time_df,freq_df, start_time, end_time)
            coherence = mc.CardiacCoherence("",ratio_coherence, start_time, end_time)
            
            
            element_time_line.append(feature)
            element_time_line.append(coherence)
            time_line.append(element_time_line)
            start_time = end_time
        return time_line
        #feature = models.feature()
        
    