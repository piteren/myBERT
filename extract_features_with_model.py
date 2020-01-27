from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os
from tqdm import tqdm

import tokenization
from bertmc import BertMC

import tensorflow as tf



# single input example object (id and 2x text)
class InputExample(object):
    """a single InputExample"""

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

# builds examples with proper counter (id)
class ExamplesBuilder:
    """builder of InputExample"""

    def __init__(self):
        self.id = 0

    # creates new example object
    def addIE(self, text_a, text_b):
        example = InputExample(self.id, text_a, text_b)
        self.id += 1
        return example


class InputFeatures(object):
  """a single set of features of data"""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids

# converts list of examples to list of `InputBatch`s.
def convert_examples_to_features(
        examples :list,                 # list of InputExample
        tokenizer,
        fitTo :None or int=     None,   # trims examples if longer than, or pads if shorter, for None fits to min(examples maxLen, safeTrim)
        safeTrim=               713):   # always trims if longer than, unless fitTo decides more

    # tokenize examples and count maxLen
    tokA = []
    tokB = []
    maxLen = 0 # counts max len of tokens @examples including separators ([CLS],[SEP],[SEP])
    for ex in examples:
        tA = tokenizer.tokenize(ex.text_a)
        cLen = len(tA) + 2
        tokA.append(tA)
        if ex.text_b:
            tB = tokenizer.tokenize(ex.text_b)
            cLen += len(tB) + 1
            tokB.append(tB)
        else: tokB.append(None)
        if cLen > maxLen: maxLen = cLen

    # trim examples if there are longer
    new_trim = fitTo if fitTo else safeTrim
    if maxLen > new_trim:
        for ix in range(len(tokA)):
            tA = tokA[ix]
            tB = tokB[ix]

            len_A = len(tA)
            len_B = len(tB) if tB else 0
            both_len = 1+len_A+1
            if len_B: both_len += len_B+1

            if both_len > new_trim:
                len_A_trg = new_trim-2
                len_B_trg = 0
                if len_B:
                    half = int((new_trim-3)/2)
                    len_A_trg = half
                    len_B_trg = half
                    if len_B < half: # B actually shorter
                        len_A_trg += half-len_B # add to A
                        len_B_trg = len_B
                    if len_A < half: # A actually shorter
                        len_A_trg = len_A
                        len_B_trg += half-len_A # add to B
                tokA[ix] = tA[:len_A_trg]
                if tB: tokB[ix] = tB[:len_B_trg]

        maxLen = new_trim # new maxLen
    # force to fit (padding)
    elif fitTo and maxLen < fitTo: maxLen = fitTo

    uidL = []
    tokL = []
    idsL = []
    mskL = []
    typL = []
    for ix in range(len(tokA)):
        tokens =            ['[CLS]']   + tokA[ix]          + ['[SEP]']
        input_type_ids =    [0]         + [0]*len(tokA[ix]) + [0]
        if tokB[ix]:
            tokens +=                   tokB[ix]            + ['[SEP]']
            input_type_ids +=           [1] * len(tokB[ix]) + [1]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        diff = maxLen-len(input_ids)
        input_ids += [0]*diff
        input_type_ids += [0]*diff
        input_mask += [0]*diff

        uidL.append(examples[ix].unique_id)
        tokL.append(tokens)
        idsL.append(input_ids)
        mskL.append(input_mask)
        typL.append(input_type_ids)

    features = [InputFeatures(*smpl) for smpl in zip(uidL,tokL,idsL,mskL,typL)]
    return features


def get_examples(
        textA :list,
        textB :None or list):
    """returns list of InputExample objects from list of texts"""

    if textB is None: textB = [None]*len(textA)
    examples = []
    exBuilder = ExamplesBuilder()
    for texts in zip(textA,textB):
        examples.append(exBuilder.addIE(*texts))
    return examples


def extract_with_model(
        model :BertMC,
        textA=                      None,
        textB=                      None,
        batch_size=                 128,
        layers_IX=                  (-1,),
        fit_seq_len : None or int=  None,  # fits num tokens of text into given length
        do_lower_case=              True,
        force16bit=                 False,
        verb=                       1):

    if verb>0: print('*** extract_with_model *** initializing...')

    model_FP = model.models_dir + '/' + model.model_name
    vocab_file = model_FP + '/vocab.txt'

    # prepare input data
    examples = get_examples(textA, textB)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    features = convert_examples_to_features(
        examples=       examples,
        tokenizer=      tokenizer,
        fitTo=          fit_seq_len)
    if verb>0: print(' > got %d samples to extract (length: %d tokens) '%(len(features),len(features[0].input_ids)))

    # pack into batches
    batch_keys = ['input_ids', 'input_mask', 'input_type_ids']
    batches = []
    batch = {key: [] for key in batch_keys}
    for feat in features:
        batch['input_ids'].append(feat.input_ids)
        batch['input_mask'].append(feat.input_mask)
        #batch['input_type_ids'].append(feat.segment_ids)
        batch['input_type_ids'].append(feat.input_type_ids)
        if len(batch['input_ids'])==batch_size:
            batches.append(batch)
            batch = {key: [] for key in batch_keys}
    if len(batch['input_ids']): batches.append(batch)

    if verb>0: print(' > starting extraction with %d batches of size %d...'%(len(batches),batch_size))

    results_lay = {ix: [] for ix in layers_IX} # list of results (np.arrs) for every layer_IX
    fetch = [model.all_encoder_layers[ix] for ix in layers_IX] # list of layer_output tensors
    if verb>0: batches = tqdm(batches)
    for batch in batches:
        feed = {model.features[key]: batch[key] for key in batch_keys}
        out = model.sess.run(fetch, feed)
        for ix in range(len(out)):
            results_lay[layers_IX[ix]].append(out[ix])

    for lay in results_lay:
        results_lay[lay] = np.concatenate(results_lay[lay],axis=0) # concatenate np.arrs of every layer into single np.arr
        n = results_lay[lay].shape[0] # number of samples
        results_lay[lay] = [np.squeeze(el) for el in  np.split(results_lay[lay],n,axis=0)] # split samples

    npt = np.float16 if force16bit else np.float32
    if force16bit: print('forced to export extract as %s'%npt)
    results = [np.concatenate([results_lay[lay][ix] for lay in layers_IX], axis=-1).astype(dtype=npt) for ix in range(len(results_lay[layers_IX[0]]))] # concatenate layers along feature dim
    if verb>0: print(' > extracted %d samples of shape %s'%(len(results), results[0].shape))
    return results


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    textA = ['It is my name.','My competence.','It is my name.','My competence.','It is my name.','My competence.',]
    textB = ['What is your name, my lord?.','My precious lord.','What is your name, my lord?.','My precious lord.','What is your name, my lord?.','My precious lord.',]

    results = extract_with_model(
        model=      BertMC('wwm_uncased_L-24_H-1024_A-16'),
        textA=      textA,
        textB=      textB,
        batch_size= 4,
        layers_IX=  (-1,-2,-3),
    )
    print(len(results))
    print(results[0].shape)

