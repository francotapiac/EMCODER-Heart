from flask import Flask, jsonify, json
import models.signal as signal


app = Flask(__name__)
@app.route("/")
def get_time_line():
    data = signal.Signal("test.txt",[], 1, 1000, "svc.joblib")
    time_line = json.dumps(data.time_line)
    return time_line
        
@app.route("/<points>/<type_signal>/<sample>")
def get_time_line_points(points, type_signal, sample):
    data = signal.Signal("",points, type_signal, sample, "svc.joblib")
    print(points)
    time_line = json.dumps(data.time_line)
    return time_line


if __name__ == "__main__":
    #get_time_line()
    app.run()


#data2 = signal2.Signal2("signal1","test.txt",1,1000,15)
#print("fin")
