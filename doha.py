import json
import csv
import sys
import logging
from models.doha import *

csv.field_size_limit(sys.maxsize)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    qa = MultiBartQA()
    if qa.args.do_train:
        qa.train()
    elif qa.args.do_eval:
        results = qa.evaluate(save_file=True)
        print(str(results))
    elif qa.args.do_generate:
        qa.generate()
    else:
        print("Specify whether to train, eval or generate")


if __name__ == '__main__':
    main()
