import configparser

config = configparser.ConfigParser()
config.read("config/testconf.cfg")
print(config.sections())
