import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import experiment as exp
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="../view/templates")
app.static_folder = "../static"

experiments = [
    exp.Experiment("Schrodinger's Car"),
    exp.Experiment("Schrodinger's Truck"),
    exp.Experiment("Schrodinger's Van")
]

@app.route("/")
def index():
    return render_template("index.html", experiments=experiments)

@app.route("/experimentView")
def experiment_view():
    experimentID = request.args.get("exp") # gets querystring ie: /experimentView?exp=EXPERIMENT-ID
    experiment = []
    for exp in experiments:
        if exp.get_name() == experimentID:
            experiment = exp
    versions = experiment.get_versions()
    return render_template("semiDetailed.html", experiment=experiment, versions=versions)

app.run()