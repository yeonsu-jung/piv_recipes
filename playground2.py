# https://docs.python.org/3/howto/logging.html#logging-basic-tutorial

import logging

if __name__ == '__main__':
    mylogger = logging.getLogger('my')
    mylogger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    mylogger.addHandler(stream_handler)

    file_handler = logging.FileHandler('my.log')
    stream_handler.setFormatter(formatter)
    mylogger.addHandler(file_handler)

    mylogger.info("server start!!!")



