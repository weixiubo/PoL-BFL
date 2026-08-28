#Here define some config used for logging
import logging
from datetime import datetime
import os
encoding = "utf-8"
level = logging.DEBUG
format = '%(asctime)s %(name)s %(levelname)s:%(message)s'

# Allow override via environment variables (prefer PoL/VeryFL specific, then default)
log_folder = os.environ.get("POL_LOG_DIR") or os.environ.get("VERYFL_LOG_DIR") or "log"
def get_file_name():
    time = datetime.strftime(datetime.now(), '%Y_%m_%d_')
    suffix = ".log"
    order = 0
    while(True):
        filename = os.path.join(log_folder,time + str(order) + suffix)
        if(os.path.exists(filename)):
            order += 1
            continue
        else:
            return filename


def set_log_config():
    # Ensure directory exists
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=get_file_name(),
        encoding=encoding,
        level=level,
        format=format
    )