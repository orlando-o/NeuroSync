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
experiments = model.get_all_exps()

@app.route("/")
def index():
    return render_template("index.html", experiments=experiments)

@app.route("/experimentView")
def experiment_view():
    experimentID = request.args.get("exp") # gets querystring ie: /experimentView?exp=EXPERIMENT-ID
    experiment = None
    for exp in experiments:
        if exp == experimentID:
            experiment = exp
    model.set_current_experiment(experiment)
    print(experiment)
    versions = model.get_versions()
    return render_template("semiDetailed.html", experiment=model.get_current_experimentID(), versions=versions)

app.run()