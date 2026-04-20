import os, logging

def get_logger(log_file=None, level=logging.INFO):
    logger = logging.getLogger('forecast')
    logger.setLevel(level)
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter('%(asctime)s[%(name)s][%(levelname)s] %(message)s')
    logger.propagate = False
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    pass  
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.propagate = False
        logger.addHandler(fh)

    return logger
