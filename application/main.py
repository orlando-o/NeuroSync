import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.experiment import Model
from flask import Flask, render_template, request
import markdown

app = Flask(__name__, template_folder="../view/templates")
app.static_folder = "../static"
model = Model()
model.set_current_experiment("SchrodingersCar"),
model.set_current_experiment("SchrodingersTruck"),
model.set_current_experiment("SchrodingersVan")
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
    return render_template("detailed.html", render_markdown = render_markdown, version=version, experiment=experiment, model=model,)

@app.route("/detailedView", methods=["POST"])
def upload_file():
    if request.files["fileUpload"]:
        file = request.files['fileUpload']
        upload_dir = os.path.join("res", "uploadedFiles")
        # upload_dir = '.\\res\\uploadedFiles'
        file.save(os.path.join(upload_dir, file.filename))
        model.generate_ml_stats(os.path.join(upload_dir, file.filename))
    version = request.args.get("vrs")
    experiment = request.args.get("exp")
    model.set_current_experiment(experiment)
    model.set_current_version(version)
    return render_template("detailed.html", version=version, experiment=experiment, model=model)
    

def render_markdown(text):
    markdown_text = text
    html_content = markdown.markdown(markdown_text)
    return html_content


@app.route("/addBranch")
def addBranch():
    version = request.args.get("vrs")
    experiment = request.args.get("exp")
    model.set_current_experiment(experiment)
    model.set_current_version(version)
    version = model.add_version(model.get_current_versionID())
    return render_template("detailed.html", version=version, experiment=experiment, model=model)
app.run()