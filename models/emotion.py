from sklearn.svm import SVC

class Emotion:
    def __init__(self, name, description, arousal, valence, dominance):
        self.id = 0
        self.name = name
        self.description = description
        self.arousal = arousal
        self.valence = valence
        self.dominance = dominance
        pass
    
    def predict_emotion(features):
        time_features = features.time_features
        freq_features = features.frequency_features
        df = time_features.append(freq_features)