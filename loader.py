import os.path


def init_logging_path(dir_log):
    dir_log = os.path.join(dir_log, f"log/")
    if os.path.exists(dir_log) and os.listdir(dir_log):
        dir_log += f"log_{len(os.listdir(dir_log)) + 1}.log"
        with open(dir_log, "w"):
            os.utime(dir_log, None)
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
        dir_log += f"log_{len(os.listdir(dir_log)) + 1}.log"
        with open(dir_log, "w"):
            os.utime(dir_log, None)
    return dir_log