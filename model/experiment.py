class Experiment:
    def __init__(self, name):
        self.name = name
        self.versions = []
    
    def add_version(self, version):
        self.versions.append(version)
    
    def get_name(self):
        return self.name
    
    def set_name(self, name):
        self.name = name

class Version:
    def __init__(self, name):
        self.name = name
        self.configs_and_stats = None
    
    def get_configs_and_stats(self, configs_and_stats):
        self.configs_and_stats = configs_and_stats
    
    def get_configs_and_stats(self):
        return self.configs_and_stats
    
    def get_name(self):
        return self.name
    
    def set_name(self, name):
        self.name = name