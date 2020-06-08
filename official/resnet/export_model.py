import tensorflow as tf
from official.resnet import resnet_model
from official.resnet.imagenet_main import ImagenetModel
import sys


def export(check_point_path, output_model_path, batch_size=None):
    tf.reset_default_graph()
    with tf.Session() as sess:
        features = tf.placeholder(tf.float32, [batch_size, 224, 224, 3], name='input_tensor')
        model = ImagenetModel(50, 'channels_last', resnet_version=1, dtype=tf.float32)
        logits = model(features, False)
        tf.contrib.quantize.experimental_create_eval_graph(
            input_graph=tf.get_default_graph(),
            weight_bits=8,
            activation_bits=8,
            symmetric=True,
            quant_delay=None,
            scope=None)
        
        saver = tf.train.Saver()
        saver.restore(sess, check_point_path)
        
        output_node_names =['resnet_model/final_dense']
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), output_node_names)
        
        with open(output_model_path, "wb") as f:
            f.write(frozen_graph.SerializeToString())
        
        print('model saved to ' + output_model_path)


if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2])

        
        
  
        
        
                                
