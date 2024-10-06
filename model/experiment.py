import csv
import google.generativeai as genai
import os
from model.gemini_api import gemini_api_key
    
hyperparameter_prompt = """
                        Give a general description of the model architecture used in this code (i.e. is the model architecure a neural network with n hidden layers, a CNN with x layers and feature maps, etc.),
                        extract the machine learning hyperparameters from this code and give them in a readable format (not a json string). This is to keep track of version control. Use a format like this, 
                        add any extra hyperparemeters that you find to the fomrat or take any away that you don't find:


                        ## Model Architecture:

                        The code implements a Multi-Layer Feedforward Neural Network (MLFN) with one hidden layer. 

                        **Structure:**

                        * **Input Layer:** 10 neurons (5 for the force values and 5 for the displacement values)
                        * **Hidden Layer:** 10 neurons with the 'Tanh' activation function.
                        * **Output Layer:** 1 neuron to predict the displacement value.

                        ## Hyperparameters:

                        The following machine learning hyperparameters are extracted from the code:

                        *Epochs: 100000
                        * **Learning Rate:** 0.1 (dynamically adjusted during training)
                        * **Optimizer:** Stochastic Gradient Descent (SGD)
                        * **Loss Function:** Mean Squared Error (MSE)
                        * **Step Size Rate:** 1.01 (used to increase the learning rate when loss decreases)
                        * **Step Back:** 0.4 (used to decrease the learning rate when loss increases)

                        **Note:** The learning rate is dynamically adjusted based on the loss function's value. If the loss decreases, the learning rate is increased by a factor of 1.01. Conversely, if the loss increases, the learning rate is decreased by a factor of 0.4.
                        """

genai.configure(api_key=gemini_api_key)
gen_model = genai.GenerativeModel("gemini-1.5-flash")

class Model:
    def __init__(self):
        self.current_experimentID = None
        self.fieldnames = ["ExpID", "VerID", "StatPath"]
    
    def set_current_experiment(self, name):
        self.current_experimentID = name
        if name not in self.get_all_exps():
            self.add_version(None)

    def get_current_experimentID(self):
        return self.current_experimentID

    def add_version(self, parent_version):
        if parent_version == None:
            VerID = "0"
        else:
            VerID = None
        with open("res/exp_data.csv", "a") as exp_data_file:
            exp_data_writer = csv.DictWriter(exp_data_file, fieldnames=self.fieldnames)
            exp_data_writer.writerow(
                {"ExpID": self.current_experimentID, "VerID": VerID, "StatPath": ""}
            )
    
    def add_stats_to_version(self, ml_file_path, VerID):
        pass
    
    def get_most_recent_child(self, parent_node):
        children_verIDs = []
        with open("res/exp_data.csv") as exp_data_file:
            exp_data_reader = csv.DictReader(exp_data_file)

            for row in exp_data_reader:
                pass
    
    def get_all_exps(self):
        exps = []
        with open("res/exp_data.csv") as exp_data_file:
            exp_data_reader = csv.DictReader(exp_data_file)
            for row in exp_data_reader:
                expID = row["ExpID"]
                if expID not in exps:
                    exps.append(expID)
            
            return exps
    
    def get_versions(self):
        versions = []
        with open("res/exp_data.csv") as exp_data_file:
            exp_data_reader = csv.DictReader(exp_data_file)
            for row in exp_data_reader:
                if row["ExpID"] == self.current_experimentID:
                    versions.append(row["VerID"])
            
            return versions #
