

class CardiacCoherence:
    def __init__(self, emotion, ratio_coherence, start_time, end_time):
        self.id = 0
        self.ratio_coherence = ratio_coherence
        self.scale = ""
        self.emotion = emotion
        self.description = self.create_description(self.ratio_coherence)
        self.alert = self.generate_alert(self.ratio_coherence)
        self.start_time = start_time
        self.end_time = end_time
        pass
    
    
    def create_description(self,ratio_coherence):
        return ""
    
    def generate_scale(self, ratio_coherence):
        if(ratio_coherence < 0.5):
            self.scale = ""
        else:
            self.alert = False
    
    def generate_alert(self, ratio_coherence):
        if(ratio_coherence < 1):
            self.alert = True
        else:
            self.alert = False
    
