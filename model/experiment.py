import csv
    

class Model:
    def __init__(self):
        self.current_experimentID = None
        self.fieldnames = ["ExpID", "VerID", "StatPath"]
    
    def set_current_experiment(self, name):
        self.current_experimentID = name
    
    def add_version(self, parent_version):
        VerID = None
        StatPath = ""
        with open("res/exp_data.csv", "a") as exp_data_file:
            exp_data_writer = csv.DictWriter(exp_data_file, fieldnames=self.fieldnames)
            exp_data_writer.writerow(
                {"ExpID": self.current_experiment, "VerID": VerID, "StatPath": ""}
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
            
            return versions
