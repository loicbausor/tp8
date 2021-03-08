# coding: utf8
import os
import argparse
import logging

# Custom Errors
class Error(Exception):
    pass

class CustomError(Error):
    pass

# Parser
parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str,
                    help="path to input file.")
parser.add_argument("--param", type=float,
                    help="example of parameter.")
parser.add_argument("--outdir", type=str,
                    help="path to output file.")

args = parser.parse_args()

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("myscript.log")
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)
c_format = logging.Formatter('%(name)s - [%(levelname)s] - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# functions
def do_stuff(condition=False):
    if condition:
        print("I can do some really nice stuff!")
    raise CustomError("cannot print nice stuff, condition is {}!".format(condition))

# main function of your script
if __name__ == "__main__":
    logger.info("starting myscript :-)")
    try:
        do_stuff()
    except (CustomError,) as e:
        logger.warning(str(e))
