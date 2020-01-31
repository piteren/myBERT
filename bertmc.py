"""

 2019 (c) piteren

"""

import tensorflow as tf

from putils.neuralmess.dev_manager import manage_dev

from modeling import BertModel, get_assignment_map_from_checkpoint, BertConfig

# Bert Model Class
class BertMC(BertModel):

    def __init__(
            self,
            model_name :str,
            models_dir=                 '_models',
            dev_id :int=                0,
            is_training=                False,
            use_one_hot_embeddings=     False,
            verb=                       0):

        self.model_name = model_name
        self.models_dir = models_dir
        if verb>0: print('\n*** BertMC *** initializing (folder: %s, model: %s)'%(self.models_dir, self.model_name))

        self.graph = tf.Graph()
        with self.graph.as_default():

            device_str = manage_dev(dev_id=dev_id, verb=verb)['devices_strL'][0]
            if verb>1: print(' > building graph on cuda: %s' % device_str)
            with tf.device(device_str):

                self.features = {
                    'input_ids':        tf.placeholder(shape=[None,None], dtype=tf.int32),
                    'input_mask':       tf.placeholder(shape=[None,None], dtype=tf.int32),
                    'input_type_ids':   tf.placeholder(shape=[None,None], dtype=tf.int32)}

                super(BertMC, self).__init__(
                    config=                 BertConfig.from_json_file(self.models_dir + '/' + self.model_name + '/bert_config.json'),
                    is_training=            is_training,
                    input_ids=              self.features['input_ids'],
                    input_mask=             self.features['input_mask'],
                    token_type_ids=         self.features['input_type_ids'],
                    use_one_hot_embeddings= use_one_hot_embeddings)

            self.tvars = tf.trainable_variables()

            checkpoint = self.models_dir + '/' + self.model_name + '/bert_model.ckpt'
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(self.tvars, checkpoint)
            tf.train.init_from_checkpoint(checkpoint, assignment_map)
            init = tf.global_variables_initializer()

        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(init)


if __name__ == "__main__":

    modelA = BertMC(model_name='uncased_L-12_H-768_A-12')