class Alert:
    
    def __init__(self, start_time, end_time):
        self.id = 0
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        
        pass