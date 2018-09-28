import logging
import inspect
import time

dashes = 15
logger = None

def basicConfig(level, filename=None, runId=None):
    global logger
    if level == 1:
        logLevel = logging.INFO
    elif level == 2:
        logLevel = logging.DEBUG
    else:
        print("Unknown log level:", str(level) + "; defaulting to INFO")
        logLevel = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = None
    if filename:
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.INFO)
        fh_d = logging.FileHandler(filename + "_debug")
        fh_d.setLevel(logging.DEBUG)

        logger.addHandler(fh)
        logger.addHandler(fh_d)

    ch = logging.StreamHandler()
    ch.setLevel(logLevel)
    logger.addHandler(ch)

    logger.info("-"*dashes + " started logging run at " + timestamp() + " " + 
            "-"*dashes)

def argsToString(args):
    return " ".join([str(arg) for arg in args])

def debug(*args):
    logger.debug(argsToString(args))

def info(*args):
    logger.info(argsToString(args))

def warning(*args):
    logger.warning(argsToString(args))

def error(*args):
    logger.warning(argsToString(args))

def critical(*args):
    logger.warning(argsToString(args))

def timestamp():
    return time.strftime("%H:%M::%D")

def run_id():
    return time.strftime("%d_%M%H")
