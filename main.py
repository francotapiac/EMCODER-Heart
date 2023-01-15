from flask import Flask, jsonify, json
import models.signal as signal

app = Flask(__name__)
@app.route("/")
def get_time_line_signal():
    print("Conectando API")
    return json.dumps({"data": "conectado"})

@app.route("/template")
def get_time_line_template():
    data = signal.Signal("test.txt",[], 1, 1000, "svc.joblib")
    time_line = data.time_line
    time_line.append({"points_signal": data.templates.tolist()})
    time_line.append({"times": data.templates_ts.tolist()})
    time_line = json.dumps(time_line)
    return time_line

@app.route("/heart/<path>/<sample>/<window>/<shif>")
def get_time_line_heart(path, sample, window, shif):
    print(shif)
    data = signal.Signal(path,[], 1, float(sample), int(window), int(shif), "svc.joblib")
    time_line = data.time_line
    time_line.append({"fci": data.heart_rate.tolist()})
    time_line.append({"times_fci": data.heart_rate_ts.tolist()})
    time_line.append({"rr":data.rr_peaks.tolist()})
    time_line.append({"signal":data.filtered_signal.tolist()})
    time_line.append({"time_signal":data.t.tolist()})

    time_line = json.dumps(time_line)

    return time_line
        
@app.route("/<points>/<type_signal>/<sample>")
def get_time_line_points(points, type_signal, sample):
    data = signal.Signal("",points, type_signal, sample, "svc.joblib")
    time_line = json.dumps(data.time_line)
    return time_line

@app.route("/signal")
def get_points_signal():
    signal = json.dumps({"signal": data.filtered_signal.tolist(), "tiempo":data.t.tolist()})
    return signal

def process_signal(path, type_signal, sample):
    return signal.Signal(path,[], type_signal, sample, "svc.joblib")

if __name__ == "__main__":
    #get_time_line()

    app.run()


#data2 = signal2.Signal2("signal1","test.txt",1,1000,15)
#print("fin")
