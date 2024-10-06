import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.experiment import Model
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="../view/templates")
app.static_folder = "../static"
model = Model()
model.set_current_experiment("Schrodinger's Car"),
model.set_current_experiment("Schrodinger's Truck"),
model.set_current_experiment("Schrodinger's Van")
experiments = None

@app.route("/")
def index():
    experiments = model.get_all_exps()
    return render_template("index.html", experiments=experiments)

@app.route("/addExperiment")
def add_experiment():
    newExperiment = request.args.get("name")
    model.set_current_experiment(newExperiment)
    experiments = model.get_all_exps()
    return render_template("index.html", experiments=experiments)

@app.route("/experimentView")
def experiment_view():
    experiments = model.get_all_exps()
    experimentID = request.args.get("exp") # gets querystring ie: /experimentView?exp=EXPERIMENT-ID
    experiment = None
    for exp in experiments:
        if exp == experimentID:
            experiment = exp
    model.set_current_experiment(experiment)
    print(experiment)
    versions = model.get_versions()
    return render_template("semiDetailed.html", experiment=model.get_current_experimentID(), versions=versions)

@app.route("/detailedView")
def detailedView():
    version = request.args.get("vrs")
    experiment = request.args.get("exp")
    model.set_current_experiment(experiment)
    model.set_current_version(version)
    return render_template("detailed.html", version=version, experiment=experiment, model=model)

@app.route("/addBranch")
def addBranch():
    version = request.args.get("vrs")
    experiment = request.args.get("exp")
    model.set_current_experiment(experiment)
    model.set_current_version(version)
    model.add_version(model.get_current_versionID())
    version = model.get_current_versionID()
    return render_template("detailed.html", version=version, experiment=experiment, model=model)
app.run()