import configparser

class Network(object):
    """Skeleton class that will eventually host networks.
    """
    def __init__(self, configpath: str) -> None:
        self.config: configparser.ConfigParser = configparser.ConfigParser()
        self.config.read(configpath)
        print(self.config._sections)

