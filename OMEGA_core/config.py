import configparser


class Preset_configurations_default:
    def __init__(self, configfile=None):
        self.config = configparser.ConfigParser()
        if configfile == None:
            self.config.read("./OMEGA_core/configurations/configuration_msm.ini")
            self.configurated = True
        else:
            self.config.read(configfile)
            self.configurated = True