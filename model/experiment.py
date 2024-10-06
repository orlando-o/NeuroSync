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
        self.current_versionID = None
        self.fieldnames = ["ExpID", "VerID", "StatPath"]
    
    def set_current_experiment(self, name):
        self.current_experimentID = name
        if name not in self.get_all_exps():
            self.add_version(None)
            os.path.mkdir(os.path.join("res", "Experiment" + self.current_experimentID))

    def get_current_experimentID(self):
        return self.current_experimentID
    
    def set_current_version(self, name):
        self.current_versionID = name
    
    def get_current_versionID(self):
        return self.current_versionID

    def add_version(self, parent_version):
        if parent_version == None:
            VerID = "0" #
        else:
            if not self.get_child_versions(parent_version):
                VerID = parent_version + ".1"
            else:
                child_numbers = []
                for child_version in self.get_child_versions(parent_version):
                    split_child_version = child_version.split(".")
                    end_number_str = split_child_version[len(split_child_version)-1:len(split_child_version)][0]
                    end_number = int(end_number_str)
                    child_numbers.append(end_number)
                VerID = parent_version + "." + str(max(child_numbers) + 1)
        with open("res/exp_data.csv", "a") as exp_data_file:
            exp_data_writer = csv.DictWriter(exp_data_file, fieldnames=self.fieldnames)
            exp_data_writer.writerow(
                {"ExpID": self.current_experimentID, "VerID": VerID, "StatPath": ""}
            )
        
        return VerID
    
    def get_child_versions(self, parent_node):
        children_verIDs = []
        with open("res/exp_data.csv") as exp_data_file:
            exp_data_reader = csv.DictReader(exp_data_file)

            for row in exp_data_reader:
                if row["ExpID"] == self.current_experimentID:
                    verID = row["VerID"]
                    if parent_node == "0" and verID != "0" and len(verID) == 1:
                        children_verIDs.append(verID)
                    
                    else:
                        split_verID = verID.split(".")
                        parent_split_verID = parent_node.split(".")

                        if len(split_verID) > len(parent_split_verID):
                            is_decendant = split_verID[:len(parent_split_verID)] == parent_split_verID
                            next_generation = len(split_verID) == len(parent_split_verID) + 1

                            if is_decendant and next_generation:
                                children_verIDs.append(verID)
        
        return children_verIDs

    def generate_ml_stats(self, ml_file_path):
        file_reader = open(ml_file_path, 'r')
        code = file_reader.read()

        prompt = hyperparameter_prompt + code
        response = gen_model.generate_content(prompt).text
        response_path = os.path.join("res", "Experiment" + self.current_experimentID, self.current_versionID + ".txt")

        with open(response_path, "w") as response_writer:
            response_writer.write(response)

        self.update_csv_with_stats(response_path)

        return response

    def get_ml_stats_path(self):
        with open("res/exp_data.csv", "r") as exp_data_file:
            exp_data_reader = csv.DictReader(exp_data_file)
            for row in exp_data_reader:
                if row["ExpID"] == self.current_experimentID and row["VerID"] == self.current_versionID:
                    ml_stats_path = row["StatPath"]
        
        return ml_stats_path
    
    def get_ml_stats(self, ml_stats_path):
        with open(ml_stats_path, "r") as ml_stats_reader:
            ml_stats = ml_stats_reader.read()

        return ml_stats
    
    def update_csv_with_stats(self, stats_path):
        updated_data = []
        with open("res/exp_data.csv", "r") as exp_data_file:
            exp_data_reader = csv.DictReader(exp_data_file)
            for row in exp_data_reader:
                if row["ExpID"] == self.current_experimentID and row["VerID"] == self.current_versionID:
                    updated_data.append(
                        {
                            "ExpID": row["ExpID"],
                            "VerID": row["VerID"],
                            "StatPath": stats_path
                        }
                    )
                else:
                    updated_data.append(
                        {
                            "ExpID": row["ExpID"],
                            "VerID": row["VerID"],
                            "StatPath": row["StatPath"]
                        }
                    )
        
        with open("res/exp_data.csv", "w") as exp_data_file:
            exp_data_writer = csv.DictWriter(exp_data_file, fieldnames=self.fieldnames)
            exp_data_writer.writeheader()
            exp_data_writer.writerows(updated_data)

    
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
            
        return versions