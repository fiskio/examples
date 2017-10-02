import torch
import os
import glob
import argparse
import math

import data
import main

parser = argparse.ArgumentParser(description='Language Model Evaluator')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--dir', type=str, default='./models/model/',
                    help='directory containing models')
args = parser.parse_args()
best_model = sorted(glob.glob(os.path.join(args.dir, '*.pt')))[-1]

corpus = data.Corpus(args.data)
test_data = main.batchify(corpus.test, 64)

with open(best_model, 'rb') as f:
    print('Loading CNTK model from path', best_model)
    model = torch.load(f)

# Run on test data.
test_loss = main.evaluate(test_data)
print('=' * 89)
print('| Model: %s | test loss {:5.2f} | test ppl {:8.2f}'.format(
    best_model, test_loss, math.exp(test_loss)))
print('=' * 89)
