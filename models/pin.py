

class Pin:
    def __init__(self, start_time, final_time, channel, content):
        self.id = 0
        self.start_time = start_time
        self.final_time = final_time
        self.duration = final_time - start_time
        self.channel = channel
        self.content = content
        
        pass