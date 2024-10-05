import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import experiment as exp
from flask import Flask, render_template, request


app = Flask(__name__)

experiments = [
    exp.Experiment("Schrodinger's Car"),
    exp.Experiment("Schrodinger's Truck"),
    exp.Experiment("Schrodinger's Van")
]

@app.route("/")
def index():
    return render_template("index.html", experiments=experiments)

app.run()