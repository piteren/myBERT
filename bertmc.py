"""

 2019 (c) piteren

"""

import tensorflow as tf

from modeling import BertModel, get_assignment_map_from_checkpoint, BertConfig

# Bert Model Class
class BertMC:

    def __init__(
            self,
            bert_config,
            device=                     '/device:GPU:0',
            checkpoint=                 None,
            is_training=                False,
            use_one_hot_embeddings=     False):

        self.graph = tf.Graph()

        with self.graph.as_default():

            with tf.device(device):

                self.features = {
                    'input_ids':        tf.placeholder(shape=[None,None], dtype=tf.int32),
                    'input_mask':       tf.placeholder(shape=[None,None], dtype=tf.int32),
                    'input_type_ids':   tf.placeholder(shape=[None,None], dtype=tf.int32)}

                self.model = BertModel(
                    config=                 bert_config,
                    is_training=            is_training,
                    input_ids=              self.features['input_ids'],
                    input_mask=             self.features['input_mask'],
                    token_type_ids=         self.features['input_type_ids'],
                    use_one_hot_embeddings= use_one_hot_embeddings)

            self.tvars = tf.trainable_variables()

        if checkpoint: self.init_from_ckpt(checkpoint)

    def init_from_ckpt(self, checkpoint):
        with self.graph.as_default():
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(self.tvars, checkpoint)
            tf.train.init_from_checkpoint(checkpoint, assignment_map)


if __name__ == "__main__":

    model_dir = '_models/uncased_L-12_H-768_A-12'
    bert_config_file = model_dir + '/bert_config.json'
    checkpoint = model_dir + '/bert_model.ckpt'
    bert_config = BertConfig.from_json_file(bert_config_file)

    modelA = BertMC(
        bert_config=    bert_config,
        checkpoint=     checkpoint)
    modelB = BertMC(
        bert_config=    bert_config,
        checkpoint=     checkpoint)
    modelC = BertMC(
        bert_config=    bert_config,
        checkpoint=     checkpoint)