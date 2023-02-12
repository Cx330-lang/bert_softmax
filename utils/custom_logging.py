# -*- encoding: utf-8 -*-
"""
@ Author: 钱朗
@ File: __init__.py
"""

import logging
import os.path
import time


def get_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)

    # 向文件输出
    haddler = logging.FileHandler(log_path, encoding="UTF-8")
    haddler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name) - %(levlname)s - %(message)s')
    haddler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    logger.addHandler(haddler)
    logger.addHandler(console)

    return logger
