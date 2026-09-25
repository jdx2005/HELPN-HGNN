import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
fh = logging.FileHandler('run.log', encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)