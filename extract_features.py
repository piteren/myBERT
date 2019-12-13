# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import os
import re

import modeling
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_string(
    "models_dir", None,
    "base directory of models")

flags.DEFINE_string(
    "bert_model_dir", None,
    "directory of current bert model")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


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


def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn


def model_fn_builder(
        bert_config,
        init_checkpoint,
        layer_indexes,
        use_tpu,
        use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        model = modeling.BertModel(
            config=                 bert_config,
            is_training=            False,
            input_ids=              features['input_ids'],
            input_mask=             features['input_mask'],
            token_type_ids=         features['input_type_ids'],
            use_one_hot_embeddings= use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT: raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        all_layers = model.get_all_encoder_layers()

        predictions = {"unique_id": features["unique_ids"]}

        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=           mode,
            predictions=    predictions,
            scaffold_fn=    scaffold_fn)
        return output_spec

    return model_fn

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

    newTrim = fitTo if fitTo else safeTrim
    # trim examples if there are longer
    if maxLen > newTrim:
        for ix in range(len(tokA)):
            tA = tokA[ix]
            tB = tokB[ix]
            bothLen = len(tA) + 2
            divBy = 1
            if tB:
                bothLen += len(tB) + 1
                divBy = 2
            # trim
            if bothLen > newTrim:
                nPads = 3 if divBy==2 else 2
                singleLen = int((newTrim-nPads)/divBy)
                tokA[ix] = tA[:singleLen]
                if tB:
                    tokB[ix] = tB[:singleLen]
        maxLen = newTrim # new maxLen
    # force to fit (to pad)
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

    """
    uidL = []
    tokL = []
    idsL = []
    mskL = []
    typL = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        #print(len(tokens_a))
        #print(tokens_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)


        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_type_ids.append(0)
            input_mask.append(0)


        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        
        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (example.unique_id))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        uidL.append(example.unique_id)
        tokL.append(tokens)
        idsL.append(input_ids)
        mskL.append(input_mask)
        typL.append(input_type_ids)
    """

    features = [InputFeatures(*smpl) for smpl in zip(uidL,tokL,idsL,mskL,typL)]
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    textA = []
    textB = []
    with tf.gfile.GFile(input_file, "r") as reader:
        while True:
            line = tokenization.convert_to_unicode(reader.readline())
            if not line: break
            line = line.strip()
            text_a = line
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is not None:
                text_a = m.group(1)
                text_b = m.group(2)
            textA.append(text_a)
            textB.append(text_b)
    return getExamples(textA,textB)


def getExamples(
        textA :list,
        textB :None or list):
    """returns list of InputExample objects from list of texts"""

    if textB is None: textB = [None]*len(textA)
    examples = []
    exBuilder = ExamplesBuilder()
    for texts in zip(textA,textB):
        examples.append(exBuilder.addIE(*texts))
    return examples


def extract(
        textA=                  None,
        textB=                  None,
        file=                   None,
        layersIX=               (-1,),
        modelsDir=              '_models',
        fitSeqLen :None or int= None): # fits num tokens of text into given length

    FLAGS.models_dir = modelsDir
    FLAGS.bert_model_dir = FLAGS.models_dir + '/uncased_L-12_H-768_A-12'
    #FLAGS.bert_model_dir = FLAGS.models_dir + '/wwm_uncased_L-24_H-1024_A-16'
    FLAGS.vocab_file = FLAGS.bert_model_dir + '/vocab.txt'
    FLAGS.bert_config_file = FLAGS.bert_model_dir + '/bert_config.json'
    FLAGS.init_checkpoint = FLAGS.bert_model_dir + '/bert_model.ckpt'
    FLAGS.do_lower_case = True

    # prepare input data
    if file: examples = read_examples(file)
    else: examples = getExamples(textA,textB)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    features = convert_examples_to_features(
        examples=       examples,
        tokenizer=      tokenizer,
        fitTo=          fitSeqLen)
    FLAGS.max_seq_length = len(features[0].input_ids)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file) ## returns config object

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=     FLAGS.master,
        tpu_config= tf.contrib.tpu.TPUConfig(
            num_shards=                     FLAGS.num_tpu_cores,
            per_host_input_for_training=    is_per_host))

    model_fn = model_fn_builder(
        bert_config=            bert_config,
        init_checkpoint=        FLAGS.init_checkpoint,
        layer_indexes=          layersIX,
        use_tpu=                FLAGS.use_tpu,
        use_one_hot_embeddings= FLAGS.use_one_hot_embeddings) # by default False

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=                FLAGS.use_tpu,
        model_fn=               model_fn,
        config=                 run_config,
        predict_batch_size=     FLAGS.batch_size)

    input_fn = input_fn_builder(
        features=               features,
        seq_length=             FLAGS.max_seq_length)

    resGO = estimator.predict( # predict returns generator object
        input_fn=               input_fn,
        yield_single_examples=  True)

    return [result for result in resGO]


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    textA = ['It is my name.','My competence.']
    textB = ['What is your name, my lord?.','My precious lord.']
    results = extract(
        textA=      textA,
        textB=      textB,
    )

    print(len(results))
    print(results[0].keys())
    layKey = [key for key in list(results[0].keys()) if 'layer' in key][0]
    print(results[0][layKey].shape)