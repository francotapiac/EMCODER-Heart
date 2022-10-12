from unittest import signals
import models.signal as ms

class Project:
    def __init__(self, name_project, name_involved, description) :
        self.id = 0
        self.name_project = name_project
        self.name_involved = name_involved
        self.description = description
        self.signals = []
        pass
    
    def create_signals(self, name_signal, path, type_signal, sampling, number_segment):
        if(self.checking_repeated_name(name_signal) == False):
            signal = ms.Signal2(name_signal, path,type_signal, sampling, number_segment)
            self.signal.append(signal)
        
        
    """"
    Verifica si el nombre de la señal no se encuentra almacenado
    @param name_singal: nombre de la señal
    @return:            Falso: no se encuentra el nombre
                        Verdadero: si se encuentra el nombre
    """
    def checking_repeated_name(self, name_signal):
        if(self.signals != []):
            for signal in self.signals:
                if(signal.name_signal == name_signal):
                    return True
        return False