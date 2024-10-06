import google.generativeai as genai
import os
from gemini_api import gemini_api_key
from model.experiment import Experiment, Version

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

def get_ml_stats_and_configs(self, file_path):
    file_reader = open(file_path, 'r')
    code = file_reader.read()

    prompt = hyperparameter_prompt + code
    response = self.gen_model.generate_content(prompt)

    return response


class Controller:
    def __init__(self):
        self.current_experiment
        self.model = model

    def add_experiment(self, name):
        self.current_experiment = name
    
    def add_version(self, parent_version, exp_name):
        if parent_version is None:
            version_number = "1"
            version = Version(version_name)
        elif not parent_version.children_versions():    # if parent version has no other children
            parent_number_split = parent_version.get_name().split(".")
            parent_end_number = int(parent_number_split[len(parent_end_number)-1:parent_end_number])
            parent_number_front = parent_number_split[:len(parent_end_number)-1]
            version_name = 
            version = Version()

        experiment = self.experiments[exp_name]
        experiment.add_version(version)
    
    def add_ml_stats_and_configs(self, file, version_name, exp_name):
        experiment = self.experiments[exp_name]
        version = experiment.get_version(version_name)
        ml_stats_and_configs = get_ml_stats_and_configs(file)
        version.set_configs_and_stats(ml_stats_and_configs)
        

# test = get_ml_stats_and_configs("/Users/jaden/PycharmProjects/CGAN/MassSpringDamper.ipynb")
# test = get_ml_stats_and_configs("/Users/jaden/PycharmProjects/CGAN/SimpleFlowCNNWithFC.ipynb")
# test = get_ml_stats_and_configs("/Users/jaden/PycharmProjects/CGAN/SimpleFlowFreezingFC2TimeInputs.ipynb")
# print(test.text)

    # def get_children_versions(self, parent_verID):
    #     with open("res/exp_data.csv") as exp_data_file:
    #         exp_data_reader = csv.DictReader(exp_data_file)
    #         children_verIDs = []

    #         for row in exp_data_reader:

    #             if row["ExpID"] == self.current_experimentID:
    #                 verID = row["VerID"]
    #                 split_verID = verID.split(".")
    #                 parent_split_verID = parent_verID.split(".")

    #                 if len(split_verID) > len(parent_split_verID):
    #                     is_decendant = split_verID[:len(parent_split_verID)] == parent_split_verID
    #                     next_generation = len(split_verID) == len(parent_split_verID) + 1

    #                     if is_decendant and next_generation:
    #                         children_verIDs.append(verID)
                    
    #                 elif len(split_verID) == len(parent_split_verID):
    #                     split_verID_end = split_verID[len(split_verID)-1:len(split_verID)][0]
    #                     parent_split_verID_end = parent_split_verID[len(parent_split_verID)-1:len(parent_split_verID)][0]

    #                     if int(split_verID_end) == int(parent_split_verID_end) + 1:
    #                         children_verIDs.append(verID)
        
    #         return children_verIDs


model = Model()
# print(model.get_all_exps())

model.set_current_experiment("1")
# print(model.get_all_exps())
# print(model.get_versions())
print(model.get_children_versions("1"))
