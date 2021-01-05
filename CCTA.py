import numpy as np
import tensorflow as tf
import transformer
import nltk

import math

from tf2_ndg_benckmarks.metrics.distinct import Distinct
import utils.taware_decoder as taware_decoder
import utils.taware_layer as taware_layer
import matplotlib.pyplot as plt
#import read_ht
def w2f(path, data):
    with open(path, 'w',encoding='utf-8') as f:
        for line in data:
            f.write(str(line))
            f.write('\n')
        f.close()

def read_data(data_path):
    enc_x = np.load('{}/enc_x.npy'.format(data_path))
    dec_y = np.load('{}/dec_y.npy'.format(data_path))
    enc_t = np.load('{}/enc_t.npy'.format(data_path))
    context_lens = np.load('{}/context_lens.npy'.format(data_path))
    enc_x_lens = np.load('{}/enc_x_lens.npy'.format(data_path))
    dec_y_lens = np.load('{}/dec_y_lens.npy'.format(data_path))
    enc_t_lens = np.load('{}/enc_t_lens.npy'.format(data_path))    
    select = np.load('{}/select.npy'.format(data_path)) 
    select_topics = np.load('{}/select_topics.npy'.format(data_path)) 
    select_topics_len = np.load('{}/select_topics_len.npy'.format(data_path)) 
    return enc_x, context_lens, enc_x_lens, enc_t, enc_t_lens, dec_y, dec_y_lens, select,select_topics, select_topics_len

class Options(object):
    '''Parameters used by the HierarchicalSeq2Seq model.'''
    def __init__(self, reverse_vocabulary, num_epochs, batch_size, learning_rate, lr_decay, max_grad_norm, beam_width, vocabulary_size,embedding_size,
                 num_hidden_layers, num_hidden_units, max_dialog_len, max_utterance_len, max_topic_len, label_len, go_index, eos_index, training_mode, 
                 num_blocks, num_heads, save_model_path, dev_data_path, embedding_file):
        super(Options, self).__init__()

        self.reverse_vocabulary = reverse_vocabulary
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.max_grad_norm = max_grad_norm
        self.beam_width = beam_width
        
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.max_dialog_len = max_dialog_len
        self.max_utterance_len = max_utterance_len
        self.max_topic_len = max_topic_len
        self.label_len = label_len
        self.go_index = go_index
        self.eos_index = eos_index
        self.training_mode = training_mode
        #self.static_embeddings = static_embeddings

        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.save_model_path = save_model_path
        self.dev_data_path = dev_data_path
        self.embedding_file = embedding_file

class HT(object):
    '''A hierarchical sequence to sequence model for multi-turn dialog generation.'''
    def __init__(self, options):
        super(HT, self).__init__()

        self.options = options

        self.build_graph()
        config = tf.ConfigProto(log_device_placement = True, allow_soft_placement = True)
        self.session = tf.Session(graph = self.graph)
        #self.writer = tf.summary.FileWriter("logs/", self.session.graph)

    def __del__(self):
        self.session.close()
        print('TensorFlow session is closed.')

    def build_graph(self):
        print('Building the TensorFlow graph...')
        opts = self.options

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('Input'):
                # turn, Sc, B
                self.context = tf.placeholder(tf.int32, shape = [opts.max_dialog_len, opts.max_utterance_len, opts.batch_size], name ='context')
                self.turn_num = tf.placeholder(tf.int32, shape = [opts.batch_size], name ='turn_num')
                self.context_lens = tf.placeholder(tf.int32, shape = [opts.max_dialog_len, opts.batch_size], name ='context_lens')

                # turn, St, B
                self.topic = tf.placeholder(tf.int32, shape = [opts.max_dialog_len, opts.max_topic_len, opts.batch_size], name ='topic')
                self.topic_len = tf.placeholder(tf.int32, shape = [opts.max_dialog_len, opts.batch_size], name ='topic_lens')

                self.response = tf.placeholder(tf.int32, [opts.batch_size, None], name ='response')
                self.response_len = tf.placeholder(tf.int32, [opts.batch_size], name ='response_len')

                self.select = tf.placeholder(tf.float32, shape=(None, opts.label_len),name='select')
                self.select_topics = tf.placeholder(tf.int32, shape=(None, opts.label_len),name='select_topics')
                self.select_topics_len = tf.placeholder(tf.int32, [opts.batch_size], name ='select_topics_len')

            with tf.variable_scope('embedding', reuse = tf.AUTO_REUSE):
                embeddings = tf.get_variable('lookup_table',
										dtype = tf.float32,
										shape = [opts.vocabulary_size, opts.embedding_size],
										initializer = tf.contrib.layers.xavier_initializer(),
                                        trainable= True)
            main = tf.strided_slice(self.response, [0, 0], [opts.batch_size, -1], [1, 1])
            self.decoder_input = tf.concat([tf.fill([opts.batch_size, 1], opts.go_index), main], 1)
            #self.decoder_input = tf.concat((tf.ones_like(self.response[:, :1])*opts.go_index, self.response[:, :-1]), -1)
            
            
            
            
            # The Context Encoder.
            with tf.variable_scope('context_encoder', reuse = tf.AUTO_REUSE):
                # Perform encoding.
                con_outputs = []
                con_final_states = []
                # turn, S, B, D
                self.context = tf.reverse(self.context, [1])
                context_embed = transformer.embedding(self.context, opts.vocabulary_size, opts.embedding_size, reuse=tf.AUTO_REUSE)
                #context_embed = tf.nn.embedding_lookup(embeddings,self.context)
                for i in range(opts.max_dialog_len):
                    # S,B,D ---> B, S,D
                    in_seq = tf.transpose(context_embed[i,:,:,:], perm=[1, 0, 2])
                    # utt_st: B,S,E
                    with tf.variable_scope('enc_cell', reuse = tf.AUTO_REUSE):
                        con_gru_cell = tf.nn.rnn_cell.GRUCell(opts.num_hidden_units)
                    ((con_fw_outputs, con_bw_outputs), (con_fw_final_state, con_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=con_gru_cell, cell_bw=con_gru_cell, inputs=in_seq, sequence_length=self.context_lens[i,:], dtype=tf.float32)
                    con_out = tf.add(con_fw_outputs, con_bw_outputs)
                    con_st = tf.add(con_fw_final_state, con_bw_final_state)
                    # B                    
                    con_final_states.append(con_st)
                    con_outputs.append(con_out)
                print('==== The length of con_outputs: ', len(con_outputs))
                print('==== The Shape of con_out: ', con_outputs[0].get_shape())
                print('==== The length of con_final_states: ', len(con_final_states))
                print('==== The Shape of con_st: ', con_final_states[0].get_shape())

                # turn, B, D
                con_rnn_input = tf.reshape(tf.stack(con_final_states), [opts.max_dialog_len, opts.batch_size, opts.num_hidden_units])
                # B, turn, D
                con_rnn_input = tf.transpose(con_rnn_input, perm=[1, 0, 2])
                with tf.variable_scope('context_cell', reuse = tf.AUTO_REUSE):
                    rnn_gru_cell = tf.nn.rnn_cell.GRUCell(opts.num_hidden_units)
                    #rnn_gru_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(opts.num_hidden_units, reuse=tf.AUTO_REUSE) for _ in range(opts.num_hidden_layers)])
                rnn_output, rnn_state = tf.nn.dynamic_rnn(cell = rnn_gru_cell, inputs = con_rnn_input, sequence_length = self.turn_num,dtype = tf.float32)
                
                print('==== The Shape of rnn_state: ', rnn_state.get_shape())
                print('==== The Shape of rnn_output: ', rnn_output.get_shape())

                
            
            # context-aware Topic Encoder
            def topic_encoder(topic_embed, utt_output):
                # topic self-attention
                t_output, _ = transformer.multihead_attention(queries=topic_embed, keys=topic_embed, 
                num_units=opts.embedding_size, num_heads= opts.num_heads,
                dropout_rate=0.3,is_training=opts.training_mode,causality=False)
                t_output = transformer.feedforward(t_output, num_units = [4 * opts.embedding_size, opts.embedding_size])
                
                # context attention
                t_output, _ = transformer.multihead_attention(queries=t_output, keys=utt_output, 
                num_units=opts.embedding_size, num_heads= opts.num_heads,
                dropout_rate=0.3,is_training=opts.training_mode,causality=False)
                t_output = transformer.feedforward(t_output, num_units = [4 * opts.embedding_size, opts.embedding_size])
                return t_output
            
            def turn_integrate(topic_output, con_state):
                # topic_output: B, St, D
                # con_state: B,D
                attn_weights = tf.matmul(topic_output, tf.expand_dims(con_state, 2))
                # attn_weights: B * S * 1
                attn_weights = tf.nn.softmax(attn_weights, axis=1)
                integration = tf.squeeze(tf.matmul(tf.transpose(topic_output, [0,2,1]), attn_weights))
                # context: B * D
                return integration
            
            with tf.variable_scope('topic_encoder', reuse = tf.AUTO_REUSE):
                # turn, S, B, D
                self.topic = tf.reverse(self.topic, [1])
                topic_embed = transformer.embedding(self.topic, opts.vocabulary_size, opts.embedding_size, reuse = tf.AUTO_REUSE)
                #topic_embed = tf.nn.embedding_lookup(embeddings,self.topic)
                topic_outputs = []
                topic_states = []
                for i in range(opts.max_dialog_len):
                    # B, Sc, D
                    in_con = con_outputs[i]
                    # each turn, corresponds to a topic representation
                    # St,B,D--->B,St,D
                    in_topic = tf.transpose(topic_embed[i,:,:,:], perm=[1, 0, 2])
                    # B, St, D
                    out_topic = topic_encoder(in_topic, in_con)
                    # B,D
                    #state_topic = tf.reduce_sum(out_topic, axis=1)
                    state_topic = turn_integrate(out_topic,rnn_output[:,i,:])

                    topic_outputs.append(out_topic)
                    topic_states.append(state_topic)
                
                print('==== The length of topic_outputs: ', len(topic_outputs))
                print('==== The Shape of topic_states: ', topic_outputs[0].get_shape())
                print('==== The length of topic_states: ', len(topic_states))
                print('==== The Shape of topic_states: ', topic_states[0].get_shape())

                # turn, B, D
                topic_rnn_input = tf.reshape(tf.stack(topic_states), [opts.max_dialog_len, opts.batch_size, opts.num_hidden_units])
                # B, turn, D
                topic_rnn_input = tf.transpose(topic_rnn_input, perm=[1, 0, 2])
                
                with tf.variable_scope('context_cell', reuse = tf.AUTO_REUSE):
                   #t_rnn_gru_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(opts.num_hidden_units, reuse=tf.AUTO_REUSE) for _ in range(opts.num_hidden_layers)])
                   t_rnn_gru_cell = tf.nn.rnn_cell.GRUCell(opts.num_hidden_units)
                
                #topic_att_mech = tf.contrib.seq2seq.LuongAttention(opts.num_hidden_units, con_rnn_input)
                #topic_cell = tf.contrib.seq2seq.AttentionWrapper(t_rnn_gru_cell, topic_att_mech, opts.num_hidden_units) 
                
                topic_rnn_output, topic_rnn_state = tf.nn.dynamic_rnn(cell = t_rnn_gru_cell, inputs = topic_rnn_input, sequence_length = self.turn_num,dtype = tf.float32)
                #print('#### The Shape of topic_rnn_state: ', topic_rnn_state.get_shape())
                
                #topic_rnn_state = topic_rnn_output[:,-1,:]
                print('==== The Shape of topic_rnn_state: ', topic_rnn_state.get_shape())
                print('==== The Shape of topic_rnn_output: ', topic_rnn_output.get_shape())        
            
            
            
            with tf.variable_scope('topic_transition', reuse = tf.AUTO_REUSE):
                
                def multilayer_perceptron(x, weight, bias):

                    layer1 = tf.add(tf.matmul(x, weight['h1']), bias['h1'])
                    layer1 = tf.nn.relu(layer1)
                    layer2 = tf.add(tf.matmul(layer1, weight['h2']), bias['h2'])
                    layer2 = tf.nn.relu(layer2)
                    layer3 = tf.add(tf.matmul(layer2, weight['h3']), bias['h3'])
                    layer3 = tf.nn.relu(layer3)
                    out_layer = tf.add(tf.matmul(layer3, weight['out']), bias['out'])
                    return out_layer

                ### MLP parameters
                weight = {
                    'h1': tf.Variable(tf.random_normal([2*opts.num_hidden_units, 1024]),name='MLP/weight_h1'),
                    'h2': tf.Variable(tf.random_normal([1024, 512]),name='MLP/weights_h2'), 
                    'h3': tf.Variable(tf.random_normal([512, 128]),name='MLP/weights_h3'), 
                    'out': tf.Variable(tf.random_normal([128, opts.label_len]),name='MLP/weights_out')
                    #'out': tf.Variable(tf.random_normal([128, 64]),name='MLP/weights_out')
                }
                bias = {
                    'h1': tf.Variable(tf.random_normal([1024]),name='MLP/bias_h1'),
                    'h2': tf.Variable(tf.random_normal([512]),name='MLP/bias_h2'), 
                    'h3': tf.Variable(tf.random_normal([128]),name='MLP/bias_h3'), 
                    'out': tf.Variable(tf.random_normal([opts.label_len]),name='MLP/bias_out')
                }

                self.MLP_input = tf.concat([topic_rnn_state, rnn_state],-1)
                self.MLP_output = multilayer_perceptron(self.MLP_input,weight,bias)
                print('==================self.MLP_output:', self.MLP_output.get_shape())
                masked_value = tf.ones_like(self.MLP_output) * (-math.pow(2, 32) + 1)
                print('==================masked_value:', masked_value.get_shape())

                mlp_mask = tf.sequence_mask(self.select_topics_len, opts.label_len)
                masked_mlp_output = tf.where(mlp_mask, self.MLP_output, masked_value)
                print('==================masked_mlp_output:', masked_mlp_output.get_shape())
                
                masked_select = tf.where(mlp_mask, self.select, masked_value)
                
                
                #self.MLP_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.select,logits = self.MLP_output))
                
                self.select_weights = tf.nn.sigmoid(tf.expand_dims(self.MLP_output, 2))
                
                def focal_loss(select, mlp_output,alpha=0.25, gamma=2):
                    sigmoid_p = tf.nn.sigmoid(mlp_output)
                    zeros = tf.zeros_like(sigmoid_p, dtype = sigmoid_p.dtype)
                    pos_p_sub = tf.where(select > zeros, select-sigmoid_p, zeros)
                    neg_p_sub = tf.where(select > zeros, zeros, sigmoid_p)
                    cross_entropy = -alpha * (pos_p_sub**gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-08, 1.0))\
                        -(1-alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0-sigmoid_p, 1e-08, 1.0))
                    return tf.reduce_mean(cross_entropy)
                
                self.MLP_loss = focal_loss(self.select, self.MLP_output)
                #self.MLP_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.select,logits = self.MLP_output))
                
                # B, label_len
                weightes = tf.nn.sigmoid(masked_mlp_output)
                print('==================weightes:', weightes.get_shape())
                top_k_indx = tf.nn.top_k(weightes, 10)[1]
                print('==================top_k_indx:', top_k_indx.get_shape())
                top_k_select_topics = tf.batch_gather(self.select_topics, top_k_indx)
                print('==================top_k_select_topics:', top_k_select_topics.get_shape())
                self.select_word = top_k_select_topics

                #self.select_weights = tf.nn.sigmoid(tf.expand_dims(masked_mlp_output, 2))
                select_topics_embed = tf.nn.embedding_lookup(embeddings,top_k_select_topics)
                selected = tf.layers.dense(select_topics_embed, opts.num_hidden_units, activation = tf.nn.relu)
            
            with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):               

                in_decode_con = rnn_output
                #in_decode_con_len = self.turn_num
                in_decode_topic = topic_rnn_output
                in_decode_select = selected
                # B,(turn+label_len),D
                #in_decode_topic = tf.concat([topic_rnn_output, selected],1)
                print('==================in_decode_con:', in_decode_con.get_shape())
                print('==================in_decode_topic:', in_decode_topic.get_shape())
                print('==================in_decode_select:', in_decode_select.get_shape())
                #in_decode_topic = selected
                #in_decode_topic_len = tf.reshape(tf.add(self.turn_num,select_len),[opts.batch_size])
                #in_decode_topic_len = select_len
                #initial_state = tf.concat([rnn_state, self.selected_state], -1)
                initial_state = rnn_state
                print('==================initial_state:', initial_state.get_shape())
                # Define the decoder cell and the output layer.
                dec_gru_cell = tf.nn.rnn_cell.GRUCell(opts.num_hidden_units)
                output_layer = tf.layers.Dense(units = opts.vocabulary_size,
                                               kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1))
                # output_layer = taware_layer.JointDenseLayer(opts.vocabulary_size, opts.vocabulary_size, name="output_projection")
                
                
                if opts.beam_width > 0 and not opts.training_mode:
                    in_decode_con = tf.contrib.seq2seq.tile_batch(in_decode_con, multiplier=opts.beam_width)
                    in_decode_topic = tf.contrib.seq2seq.tile_batch(in_decode_topic, multiplier=opts.beam_width)
                    in_decode_select = tf.contrib.seq2seq.tile_batch(in_decode_select, multiplier=opts.beam_width)
                    initial_state = tf.contrib.seq2seq.tile_batch(initial_state, multiplier=opts.beam_width)
                    #in_decode_topic_len = tf.contrib.seq2seq.tile_batch(in_decode_topic_len, multiplier=opts.beam_width)
                    #in_decode_con_len = tf.contrib.seq2seq.tile_batch(in_decode_con_len, multiplier=opts.beam_width)
                
                con_attention = tf.contrib.seq2seq.LuongAttention(num_units = opts.num_hidden_units, memory = in_decode_con)
                topic_attention = tf.contrib.seq2seq.LuongAttention(num_units = opts.num_hidden_units, memory = in_decode_topic)
                select_attention = tf.contrib.seq2seq.LuongAttention(num_units = opts.num_hidden_units, memory = in_decode_select)
                
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(dec_gru_cell,attention_mechanism=(con_attention, topic_attention,select_attention),\
                    attention_layer_size=(opts.num_hidden_units, opts.num_hidden_units,opts.num_hidden_units), alignment_history= True, name="joint_attention")                
                
                
                
                # Perform training decoding.                
                if opts.training_mode:                    
                    training_helper = tf.contrib.seq2seq.TrainingHelper(
                        inputs = tf.nn.embedding_lookup(embeddings,self.decoder_input),
                        sequence_length = self.response_len)
                    
                    # training_decoder = taware_decoder.ConservativeBasicDecoder(decoder_cell,training_helper,\
                    #     decoder_cell.zero_state(opts.batch_size, tf.float32).clone(cell_state=initial_state), output_layer)

                    training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell = decoder_cell,
                    helper = training_helper,
                    initial_state = decoder_cell.zero_state(opts.batch_size, tf.float32).clone(cell_state=initial_state),
                    output_layer = output_layer)

                    # Dynamic decoding
                    
                    training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder = training_decoder,
                        impute_finished = True,
                        maximum_iterations = 2*tf.reduce_max(self.response_len),
                        swap_memory=True)
                    
                    self.training_logits = training_decoder_output.rnn_output
                    print('==================training_logits:', self.training_logits.get_shape())

                    predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding = embeddings,\
                            start_tokens = tf.tile(tf.constant([opts.go_index], dtype=tf.int32), [opts.batch_size]),end_token = opts.eos_index)
                    
                    # predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,helper = predicting_helper,\
                    #     initial_state = decoder_cell.zero_state(opts.batch_size, tf.float32).clone(cell_state=initial_state),output_layer = output_layer)
                    predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,helper = predicting_helper,\
                        initial_state = decoder_cell.zero_state(opts.batch_size, tf.float32).clone(cell_state=initial_state),output_layer = output_layer)
                    
                    predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = predicting_decoder,\
                        impute_finished = False, maximum_iterations = 2*tf.reduce_max(self.response_len))
                    
                    self.predicting_ids = predicting_decoder_output.sample_id
                
                    # Compute loss function.
                    current_ts = tf.to_int32(tf.minimum(tf.shape(self.response)[-1], tf.shape(self.training_logits)[1]))
                    # # 对 output_out 进行截取
                    new_response = tf.slice(self.response, begin=[0, 0], size=[-1, current_ts])

                    masks = tf.sequence_mask(self.response_len, dtype=tf.float32)
                    
                    self.dec_loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits, targets = new_response, weights = masks)
                    
                    
                    
                    # # Uncertainty
                    # # sigma_dec = log(sigma^2)
                    # self.sigma_dec = tf.get_variable('sigma_decode_loss',
					# 					dtype = tf.float32,
					# 					shape = [],
					# 					initializer = tf.initializers.random_uniform(minval=0.2, maxval=1),
                    #                     trainable= True)
                    # self.sigma_mlp = tf.get_variable('sigma_MLP_loss',
					# 					dtype = tf.float32,
					# 					shape = [],
					# 					initializer = tf.initializers.random_uniform(minval=0.2, maxval=1),
                    #                     trainable= True)
                    # # (1/sigma^2)*self.dec_loss + log(sigma)
                    # dec_loss = tf.add(tf.multiply(tf.exp(tf.negative(self.sigma_dec)),self.dec_loss), tf.multiply(0.5,self.sigma_dec))
                    # mlp_loss = tf.add(tf.multiply(tf.exp(tf.negative(self.sigma_mlp)),self.MLP_loss), tf.multiply(0.5,self.sigma_mlp))
                    
                    # # dec_loss = tf.add(tf.multiply(tf.div(1.0, self.sigma_dec), self.dec_loss), tf.div(tf.log(self.sigma_dec),2))
                    # # mlp_loss = tf.add(tf.multiply(tf.div(1.0, self.sigma_mlp), self.MLP_loss), tf.div(tf.log(self.sigma_mlp),2))
                    
                    #self.loss = self.dec_loss 
                    self.loss = self.dec_loss + 1.75*self.MLP_loss
                    #self.loss = dec_loss + mlp_loss
                    


                    
                    
                    self.params = tf.trainable_variables()

                    self.lr = tf.Variable(float(opts.learning_rate), trainable=True, dtype=tf.float32)

                    self.lr_decay_op = self.lr.assign(self.lr * opts.lr_decay)
                    
                    self.global_step = tf.Variable(0, trainable=False)
                    #self.lr = tf.train.exponential_decay(opts.learning_rate, self.global_step, 1000, opts.lr_decay)

                    
                    
                    #self.optimizer = tf.train.AdamOptimizer(learning_rate =opts.learning_rate, beta1 = 0.9, beta2 = 0.98, epsilon = 1e-8)
                    #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = opts.learning_rate)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate =self.lr)
                    
                    #gradients = self.optimizer.compute_gradients(self.loss)
                    gradients = tf.gradients(self.loss, self.params)
                    #clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
                    clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,opts.max_grad_norm)

                    #self.train_op = self.optimizer.apply_gradients(clipped_gradients)
                    self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)


                   

                else:
                    if opts.beam_width > 0:
                        predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, 
                                                                            embedding=embeddings,
                                                                            start_tokens=tf.tile(tf.constant([opts.go_index], dtype=tf.int32),[opts.batch_size]), 
                                                                            end_token=opts.eos_index,
                                                                            initial_state=decoder_cell.zero_state(opts.batch_size*opts.beam_width, tf.float32).clone(cell_state=initial_state),
                                                                            beam_width=opts.beam_width,
                                                                            output_layer=output_layer)
                    
                    else:
                        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding = embeddings,\
                            start_tokens = tf.tile(tf.constant([opts.go_index], dtype=tf.int32), [opts.batch_size]),end_token = opts.eos_index)
                    
                        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,helper = predicting_helper,\
                            initial_state = decoder_cell.zero_state(opts.batch_size, tf.float32).clone(cell_state=initial_state),output_layer = output_layer)
                        #predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,helper = predicting_helper,
                        # initial_state = decoder_cell.zero_state(opts.batch_size, tf.float32).clone(cell_state=initial_state),output_layer = output_layer)

                    
                    predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = predicting_decoder,\
                        impute_finished = False, maximum_iterations = 2*tf.reduce_max(self.response_len))
                    #self.training_logits = training_decoder_output.rnn_output
                    if opts.beam_width > 0:
                        self.predicting_ids = predicting_decoder_output.predicted_ids[:,:,0]
                    else:
                        self.predicting_ids = predicting_decoder_output.sample_id

                
                    
                    
                

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=5)
                

                
            
    def init_tf_vars(self):
        self.session.run(self.init)
        print('TensorFlow variables initialized.')
    
    def bleu(self, ref, pre):
        ref = ref.split(' ')
        pre = pre.split(' ')
        bleu_1 = nltk.translate.bleu_score.sentence_bleu([ref],pre,(1, 0, 0, 0),nltk.translate.bleu_score.SmoothingFunction().method1)
        bleu_2 = nltk.translate.bleu_score.sentence_bleu([ref],pre,(0.5,0.5),nltk.translate.bleu_score.SmoothingFunction().method1)
        bleu_3 = nltk.translate.bleu_score.sentence_bleu([ref],pre,(0.333,0.333,0.333),nltk.translate.bleu_score.SmoothingFunction().method1)
        bleu_4 = nltk.translate.bleu_score.sentence_bleu([ref],pre,(0.25, 0.25, 0.25, 0.25),nltk.translate.bleu_score.SmoothingFunction().method1)
        return bleu_1, bleu_2, bleu_3, bleu_4

    def distinct_1(self, lines):
        '''Computes the number of distinct words divided by the total number of words.

        Input:
        lines: List String.
        
        '''
        words = ' '.join(lines).split(' ')
        #if '<EOS>' in words:
        #    words.remove('<EOS>')
        num_distinct_words = len(set(words))
        return float(num_distinct_words) / len(words)


    def distinct_2(self, lines):
        '''Computes the number of distinct bigrams divided by the total number of words.

        Input:
        lines: List of strings.
        '''
        all_bigrams = []
        num_words = 0

        for line in lines:
            line_list = line.split(' ')
            #if '<EOS>' in line_list:
            #    line_list.remove('<EOS>')            
            num_words += len(line_list)
            bigrams = zip(line_list, line_list[1:])
            all_bigrams.extend(list(bigrams))

        return len(set(all_bigrams)) / float(num_words)
    


    # ========== Our own embedding-based metric ========== #
    def cal_vector_extrema(self,x, y, dic):
        # x and y are the list of the words
        # dic is the gensim model which holds 300 the google news word2ved model
        def vecterize(p):
            vectors = []
            for w in p:
                if w in dic:
                    #vectors.append(dic[w.lower()])
                    vectors.append(dic[w])
            if not vectors:
                vectors.append(np.random.randn(300))
            return np.stack(vectors)
        x = vecterize(x)
        y = vecterize(y)
        vec_x = np.max(x, axis=0)
        vec_y = np.max(y, axis=0)
        assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
        zero_list = np.zeros(len(vec_x))
        if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
            return float(1) if vec_x.all() == vec_y.all() else float(0)
        res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        return cos


    def cal_embedding_average(self, x, y, dic):
        # x and y are the list of the words
        def vecterize(p):
            vectors = []
            for w in p:
                if w in dic:
                    #vectors.append(dic[w.lower()])
                    vectors.append(dic[w])
            if not vectors:
                vectors.append(np.random.randn(300))
            return np.stack(vectors)
        x = vecterize(x)
        y = vecterize(y)
        
        vec_x = np.array([0 for _ in range(len(x[0]))])
        for x_v in x:
            x_v = np.array(x_v)
            vec_x = np.add(x_v, vec_x)
        vec_x = vec_x / math.sqrt(sum(np.square(vec_x)))
        
        vec_y = np.array([0 for _ in range(len(y[0]))])
        #print(len(vec_y))
        for y_v in y:
            y_v = np.array(y_v)
            vec_y = np.add(y_v, vec_y)
        vec_y = vec_y / math.sqrt(sum(np.square(vec_y)))
        
        assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
        
        zero_list = np.array([0 for _ in range(len(vec_x))])
        if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
            return float(1) if vec_x.all() == vec_y.all() else float(0)
        
        vec_x = np.mat(vec_x)
        vec_y = np.mat(vec_y)
        num = float(vec_x * vec_y.T)
        denom = np.linalg.norm(vec_x) * np.linalg.norm(vec_y)
        cos = num / denom
        
        # res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
        # cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        
        return cos


    def cal_greedy_matching(self, x, y, dic):
        # x and y are the list of words
        def vecterize(p):
            vectors = []
            for w in p:
                if w in dic:
                    #vectors.append(dic[w.lower()])
                    vectors.append(dic[w])
            if not vectors:
                #vectors.append(np.random.randn(300))
                vectors.append(np.zeros(300))
            return np.stack(vectors)
        x = vecterize(x)
        y = vecterize(y)
        
        len_x = len(x)
        len_y = len(y)
        
        cosine = []
        sum_x = 0 

        for x_v in x:
            for y_v in y:
                assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
                zero_list = np.zeros(len(x_v))

                if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
                    if x_v.all() == y_v.all():
                        cos = float(1)
                    else:
                        cos = float(0)
                else:
                    # method 1
                    res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
                    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

                cosine.append(cos)
            if cosine:
                sum_x += max(cosine)
                cosine = []

        sum_x = sum_x / len_x
        cosine = []

        sum_y = 0

        for y_v in y:

            for x_v in x:
                assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
                zero_list = np.zeros(len(y_v))

                if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
                    if (x_v == y_v).all():
                        cos = float(1)
                    else:
                        cos = float(0)
                else:
                    # method 1
                    res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
                    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

                cosine.append(cos)

            if cosine:
                sum_y += max(cosine)
                cosine = []

        sum_y = sum_y / len_y
        score = (sum_x + sum_y) / 2
        return score


    def cal_greedy_matching_matrix(self, x, y, dic):
        # x and y are the list of words
        def vecterize(p):
            vectors = []
            for w in p:
                if w in dic:
                    vectors.append(dic[w])
            if not vectors:
                vectors.append(np.random.randn(300))
            return np.stack(vectors)
        x = vecterize(x)     # [x, 300]
        y = vecterize(y)     # [y, 300]
        
        len_x = len(x)
        len_y = len(y)
        
        matrix = np.dot(x, y.T)    # [x, y]
        matrix = matrix / np.linalg.norm(x, axis=1, keepdims=True)    # [x, 1]
        matrix = matrix / np.linalg.norm(y, axis=1).reshape(1, -1)    # [1, y]
        
        x_matrix_max = np.mean(np.max(matrix, axis=1))    # [x]
        y_matrix_max = np.mean(np.max(matrix, axis=0))    # [y]
        
        return (x_matrix_max + y_matrix_max) / 2


    def cal_embedding_metric(self, refs, pres, w2v):    
        if len(refs) != len(pres):
            print('Shape Error!')
    
        greedys = []
        averages = []
        extremas = []
        for i in range(len(refs)):
            x = refs[i].strip().split(' ')
            y = pres[i].strip().split(' ')
            greedys.append(self.cal_greedy_matching_matrix(x, y, w2v))
            averages.append(self.cal_embedding_average(x, y, w2v))
            extremas.append(self.cal_vector_extrema(x, y, w2v))
        
        
        return np.mean(np.asarray(averages)), np.mean(np.asarray(extremas)), np.mean(np.asarray(greedys))
    

    def eval(self, pre_lines,tar_lines):
        opts = self.options
        #w2v = 
        # BLEU
        bleu_1 = []
        bleu_2 = []
        bleu_3 = []
        bleu_4 = []
        for i in range(len(pre_lines)):            
            one, two, three, four = self.bleu(tar_lines[i],pre_lines[i])
            bleu_1.append(one)
            bleu_2.append(two)
            bleu_3.append(three)
            bleu_4.append(four)
        avg_bleu_1 = sum(bleu_1)/len(bleu_1)
        avg_bleu_2 = sum(bleu_2)/len(bleu_2)
        avg_bleu_3 = sum(bleu_3)/len(bleu_3)
        avg_bleu_4 = sum(bleu_4)/len(bleu_4)

        # Distinct-1
        dis = Distinct()
        dis_1 = dis.sentence_score(pre_lines, 1)
        # Distinct-2
        dis_2 = dis.sentence_score(pre_lines, 2)
        #Embedding
        #average, extrema, greedy = self.cal_embedding_metric(tar_lines, pre_lines, opts.embedding_file)
        average, extrema, greedy = 0,0,0
        return avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4, dis_1, dis_2, average, extrema, greedy


    def train(self, enc_x, dec_y, enc_t, dialog_lens, enc_x_lens, dec_y_lens, enc_t_lens, select, select_topics, select_topics_len):
        print('Start to train the model...')
        opts = self.options
        num_examples = enc_x.shape[0]
        num_batches = num_examples // opts.batch_size
        valid_time = 0
        w_dec = []
        w_mlp = []
        epoch_loss = []
        epoch_dec_loss = []
        epoch_mlp_loss = []
        train_loss = []
        train_dec_loss = []
        train_mlp_loss = []
        dev_dec_loss =[]
        dev_mlp_loss = []
        dev_loss = []
        
        patient = 3
        fail_time = 0
        previous_losses = [1e18]*5
        for epoch in range(opts.num_epochs):
            e_loss = 0
            e_dec_loss = 0
            e_mlp_loss = 0
            perm_indices = np.random.permutation(range(num_examples))
        
            print('================', 'Epoch ', epoch+1, '================')
                    
            for batch in range(num_batches):
                #if fail_time > patient:
                #    break
                s = batch * opts.batch_size
                t = s + opts.batch_size
                batch_indices = perm_indices[s:t]
                # B turn S--turn,S,B
                batch_enc_x = np.transpose(enc_x[batch_indices,:,:], [1, 2, 0])
                batch_dialog_lens = dialog_lens[batch_indices]
                batch_enc_x_lens = np.transpose(enc_x_lens[batch_indices,:])
                batch_enc_t = np.transpose(enc_t[batch_indices,:,:], [1, 2, 0])
                #enc_t[s:t,:]
                batch_enc_t_lens = np.transpose(enc_t_lens[batch_indices,:])
                #enc_t_lens[s:t]
                batch_dec_y = dec_y[batch_indices,:]
                batch_dec_y_lens = dec_y_lens[batch_indices]
                batch_select = select[batch_indices,:]
                batch_select_topics = select_topics[batch_indices,:]
                batch_select_topics_len = select_topics_len[batch_indices]

                feed_dict = {self.context: batch_enc_x,
                             self.turn_num: batch_dialog_lens,
                             self.context_lens: batch_enc_x_lens,
                             self.topic: batch_enc_t,
                             self.topic_len: batch_enc_t_lens,
                             self.response: batch_dec_y,
                             self.response_len: batch_dec_y_lens,
                             self.select: batch_select,
                             self.select_topics: batch_select_topics,
                             self.select_topics_len: batch_select_topics_len}
                # _, loss_val, loss_dec, loss_mlp, sigma_dec, sigma_mlp,select_word = self.session.run([self.train_op, self.loss, self.dec_loss, self.MLP_loss,\
                #     self.sigma_dec, self.sigma_mlp, self.select_word], feed_dict = feed_dict)
                _, loss_val, loss_dec, loss_mlp, select_word, l_r = self.session.run([self.train_op, self.loss, self.dec_loss, self.MLP_loss,\
                   self.select_word,self.lr], feed_dict = feed_dict)      
                e_loss += loss_val
                e_dec_loss += loss_dec
                e_mlp_loss += loss_mlp

                train_loss.append(loss_val)
                train_dec_loss.append(loss_dec)
                train_mlp_loss.append(loss_mlp)

                if batch%200==0:
                    if loss_val > max(previous_losses):
                        self.session.run(self.lr_decay_op)
                    previous_losses = previous_losses[1:]+[loss_val]
                    print(previous_losses)                    
                    
                    # print loss and predicted sentences
                    print_predicted= self.session.run(self.predicting_ids, feed_dict= feed_dict)
                    print('------------------------------------------------')
                    print('Epoch {:03d} batch {:04d}/{:04d} Learning Rate {}'.format(epoch + 1, batch + 1,num_batches, l_r), flush = True)
                    #print('Current Avg Training Loss: {}'.format(sum(loss)/len(loss)))                    
                    
                    print('Overall loss: {}, dec_loss: {}, mlp_loss: {}'.format(loss_val, loss_dec, loss_mlp), flush = True)
                    #print('sigma_dec: {}, sigma_mlp: {}'.format(sigma_dec, sigma_mlp), flush = True)
                    
                    print('----------------  Samples  ----------------')
                    print('Last CONTEXT:',' '.join([opts.reverse_vocabulary[n] for n in batch_enc_x[batch_dialog_lens[0]-1,:,0] if n not in [0,1,2]]))                    
                    print('TOPIC:',' '.join([opts.reverse_vocabulary[n] for n in batch_enc_t[batch_dialog_lens[0]-1,:,0] if n not in [0,1,2]]))
                    print('Context topic: ', ' '.join([opts.reverse_vocabulary[n] for n in batch_select_topics[0] if n not in [0,1,2]]))
                    print('Select topic: ',' '.join([opts.reverse_vocabulary[n] for n in select_word[0] if n not in [0,1,2]]))
                    flag = True
                    pre = []
                    for n in print_predicted[0]: 
                        if flag:
                            if n==2:
                                flag = False
                            else:
                                if n not in [0, 1]:
                                    pre.append(opts.reverse_vocabulary[n])
                        else:
                            pass      
                    
                    
                    pre_sen = ' '.join(pre)
                    ref_sen = ' '.join([opts.reverse_vocabulary[n] for n in batch_dec_y[0] if n not in [0,1,2]])
                    print('----------------')
                    print('REF:',ref_sen)
                    print('PRE:',pre_sen)
                    print()
                    
                    
                if batch%500==0 and batch != 0:
                    # validate and save model
                    print('********************************')
                    # self.saver.save(self.session, opts.save_model_path + '/model_epoch_%02d_batch_%d' % (epoch+1, batch))
                    # print('Model saved!!!')
                    valid_time += 1
                    print('The {:03d} Validation Time. Epoch {:03d} Batch {:03d}'.format(valid_time, epoch+1, batch+1))                    
                    pre_lines, ref_lines, select_lines, avg_loss, avg_dec_loss, avg_mlp_loss,\
                        avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4, dis_1, dis_2, average, extrema, greedy = self.valid(opts.dev_data_path, valid_time)
                    
                    # if valid_time > 2 and avg_dec_loss > dev_dec_loss[-1]:
                    #     print('Smaller Dec loss. NOT a good model!')
                    #     fail_time = fail_time + 1
                    # else:
                    #     fail_time = 0
                    self.saver.save(self.session, opts.save_model_path + '/model_epoch_%02d' % (epoch+1))
                    print('Model saved!')
                    
                    with open(opts.save_model_path + '-'+str(valid_time) + '-result.txt', 'w', encoding ='utf-8') as f:
                        for i in range(len(ref_lines)):
                            f.write('ref: ' + ref_lines[i])
                            f.write('\n')
                            f.write('select: ' + select_lines[i])
                            f.write('\n')
                            f.write('pre: ' + pre_lines[i])
                            f.write('\n')                            
                            f.write('\n')
                        f.close()                   
                    
                    
                    dev_loss.append(avg_loss)
                    dev_dec_loss.append(avg_dec_loss)
                    dev_mlp_loss.append(avg_mlp_loss)
                    
                    #if dis_1<max(dev_dis1) and dis_2<max(dev_dis2) and avg_bleu_4<max(dev_bleu4):
                    #if average < max(dev_average) and extrema < max(dev_extrema) and greedy < max(dev_greedy):
                    
                    print('********************************')
                    print('')

                    
            print('********************************')           
            print('Epoch {:03d} Avg. Loss {} Avg Dec Loss: {} MLP Loss: {}'.format(epoch + 1, e_loss/num_batches,\
                e_dec_loss/num_batches,e_mlp_loss/num_batches), flush = True)
            print('********************************')
            print('')
            epoch_loss.append(e_loss/num_batches)
            epoch_dec_loss.append(e_dec_loss/num_batches)
            epoch_mlp_loss.append(e_mlp_loss/num_batches)

            w2f(opts.save_model_path+'/train_loss.txt', train_loss)
            w2f(opts.save_model_path+'/train_dec_loss.txt', train_dec_loss)
            w2f(opts.save_model_path+'/train_mlp_loss.txt', train_mlp_loss)
            w2f(opts.save_model_path+'/dev_loss.txt', dev_loss)
            w2f(opts.save_model_path+'/dev_dec_loss.txt', dev_dec_loss)
            w2f(opts.save_model_path+'/dev_mlp_loss.txt', dev_mlp_loss)
            w2f(opts.save_model_path+'/epoch_dec_loss.txt', epoch_dec_loss)
            w2f(opts.save_model_path+'/epoch_mlp_loss.txt', epoch_mlp_loss)
            w2f(opts.save_model_path+'/epoch_loss.txt', epoch_loss)
        # np.savetxt(opts.save_model_path+'/sigma_dec.txt', w_dec)
        # np.savetxt(opts.save_model_path+'/sigma_mlp.txt', w_mlp)    
        # np.savetxt(opts.save_model_path+'/train_loss.txt', train_loss)
        # np.savetxt(opts.save_model_path+'/train_dec_loss.txt', train_dec_loss)
        # np.savetxt(opts.save_model_path+'/train_mlp_loss.txt', train_mlp_loss)
        # np.savetxt(opts.save_model_path+'/dev_loss.txt', dev_loss)
        # np.savetxt(opts.save_model_path+'/dev_dec_loss.txt', dev_dec_loss)
        # np.savetxt(opts.save_model_path+'/dev_mlp_loss.txt', dev_mlp_loss)
        # np.savetxt(opts.save_model_path+'/dev_bleu1.txt', dev_bleu1)
        # np.savetxt(opts.save_model_path+'/dev_bleu2.txt', dev_bleu2)
        # np.savetxt(opts.save_model_path+'/dev_bleu3.txt', dev_bleu3)            
        # np.savetxt(opts.save_model_path+'/dev_bleu4.txt', dev_bleu4)
        # np.savetxt(opts.save_model_path+'/dev_dis1.txt', dev_dis1)
        # np.savetxt(opts.save_model_path+'/dev_dis2.txt', dev_dis2)
        # np.savetxt(opts.save_model_path+'/dev_average.txt', dev_average)
        # np.savetxt(opts.save_model_path+'/dev_extrema.txt', dev_extrema)
        # np.savetxt(opts.save_model_path+'/dev_greedy.txt', dev_greedy)

        # np.savetxt(opts.save_model_path+'/epoch_dec_loss.txt', epoch_dec_loss)
        # np.savetxt(opts.save_model_path+'/epoch_mlp_loss.txt', epoch_mlp_loss)
        # np.savetxt(opts.save_model_path+'/epoch_loss.txt', epoch_loss)
        
        print('==============Training Summary==================')
        print('Training times: ', len(train_loss))
        print('Max training Loss: ', max(train_loss))
        print('Min training Loss: ', min(train_loss))
        print('Max training Dec_loss: ', max(train_dec_loss))
        print('Min training Dec_loss: ', min(train_dec_loss))
        print('Max training MLP_loss: ', max(train_mlp_loss))
        print('Min training MLP_loss: ', min(train_mlp_loss))
        print('==============Validation Summary==================')
        print('Validation times: ', valid_time)
        print('Min dev loss: ', min(dev_loss))
        print('Min dev Decoder loss: ', min(dev_dec_loss))
        print('Min dev MLP loss: ', min(dev_mlp_loss))
        #plt.plot(train_dec_loss)
        plt.plot(epoch_dec_loss)
        # print('Max dev BLEU-1: ', max(dev_bleu1))
        # print('Max dev BLEU-2: ', max(dev_bleu2))
        # print('Max dev BLEU-3: ', max(dev_bleu3))
        # print('Max dev BLEU-4: ', max(dev_bleu4))
        # print('Max dev Distinct-1: ', max(dev_dis1))
        # print('Max dev Distinct-2: ', max(dev_dis2))
        # print('Max dev Average: ', max(dev_average))
        # print('Max dev Extrema: ', max(dev_extrema))
        # print('Max dev Greedy: ', max(dev_greedy))

        print('')
        print('')
        

    def save(self, save_path):
        print('Saving the trained model...')
        self.saver.save(self.session, save_path)

    def restore(self, restore_path):
        print('Restoring from a pre-trained model...')        
        self.saver.restore(self.session, tf.train.latest_checkpoint(restore_path))
    
    
    
    def valid(self, data_path, valid_time):                          
        print('Start Validate the model......')
        enc_x, context_lens, enc_x_lens, enc_t, enc_t_lens, dec_y, dec_y_lens, select, select_topics, select_topics_len = read_data(data_path)    
        opts = self.options
        num_examples = enc_x.shape[0]
        num_batches = num_examples // opts.batch_size        
        perm_indices = np.random.permutation(range(num_examples))
        loss_values = []
        dec_losses = []
        mlp_losses = []
        ref_lines = []
        pre_lines = []  
        select_lines = []          
        for batch in range(num_batches):
            s = batch * opts.batch_size
            t = s + opts.batch_size
            batch_enc_x = np.transpose(enc_x[s:t,:,:], [1, 2, 0])
            batch_dialog_lens = context_lens[s:t]
            batch_enc_x_lens = np.transpose(enc_x_lens[s:t,:])
            batch_enc_t = np.transpose(enc_t[s:t,:,:], [1, 2, 0])
            #enc_t[s:t,:]
            batch_enc_t_lens = np.transpose(enc_t_lens[s:t,:])
            #enc_t_lens[s:t]
            batch_dec_y = dec_y[s:t,:]
            batch_dec_y_lens = dec_y_lens[s:t]
            batch_select = select[s:t,:]
            batch_select_topics = select_topics[s:t,:]
            batch_select_topics_len = select_topics_len[s:t]
            

            feed_dict = {self.context: batch_enc_x,
                            self.turn_num: batch_dialog_lens,
                            self.context_lens: batch_enc_x_lens,
                            self.topic: batch_enc_t,
                            self.topic_len: batch_enc_t_lens,
                            self.response: batch_dec_y,
                            self.response_len: batch_dec_y_lens,
                            self.select: batch_select,
                            self.select_topics: batch_select_topics,
                            self.select_topics_len: batch_select_topics_len}
            loss_val,dec_loss,mlp_loss,predicted,select_word = self.session.run([self.loss,self.dec_loss,self.MLP_loss, self.predicting_ids,self.select_word], feed_dict = feed_dict)
            loss_values.append(loss_val)
            dec_losses.append(dec_loss)
            mlp_losses.append(mlp_loss)
            for i in range(opts.batch_size):
                
                ref = ' '.join([opts.reverse_vocabulary[n] for n in batch_dec_y[i] if n not in[0,1,2]])
                s_words = ' '.join([opts.reverse_vocabulary[n] for n in select_word[i] if n not in[0,1,2]])
                flag = True
                pre = []
                for n in predicted[i]: 
                    if flag:
                        if n==2:
                            flag = False
                        else:
                            if n not in [0,1]:
                                pre.append(opts.reverse_vocabulary[n])
                    else:
                        pass      

                #pre = ' '.join([opts.reverse_vocabulary[n] for n in predicted[i] if n not in[0,1,2]])
                pre = ' '.join(pre)
                ref_lines.append(ref)
                pre_lines.append(pre)
                select_lines.append(s_words)
        avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4, dis_1, dis_2, average, extrema, greedy = self.eval(pre_lines,ref_lines)
        print('Loss: {}, Dec_loss: {}, MLP_loss: {}'.format(sum(loss_values)/len(loss_values),sum(dec_losses)/len(dec_losses),sum(mlp_losses)/len(mlp_losses)))
        print('Valid BLEU: {}, {}, {}, {}'.format(avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4))
        # print('Valid Embedding: {}, {}, {}'.format(average, extrema, greedy))        
        print('Valid DISTINCT: {}, {}'.format(dis_1, dis_2)) 
        return pre_lines, ref_lines, select_lines, sum(loss_values)/len(loss_values),sum(dec_losses)/len(dec_losses),sum(mlp_losses)/len(mlp_losses), avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4, dis_1, dis_2, average, extrema, greedy

    def predict(self, data_path): 
        print('=======================================')       
        print('start TEST the model......')
        enc_x, context_lens, enc_x_lens, enc_t, enc_t_lens, dec_y, dec_y_lens, select, select_topics, select_topics_len = read_data(data_path)    
        opts = self.options
        num_examples = enc_x.shape[0]
        num_batches = num_examples // opts.batch_size        
        ref_lines = []
        pre_lines = []  
        test_samples = []          
        for batch in range(num_batches):
            s = batch * opts.batch_size
            t = s + opts.batch_size
            batch_enc_x = np.transpose(enc_x[s:t,:,:], [1, 2, 0])
            batch_dialog_lens = context_lens[s:t]
            batch_enc_x_lens = np.transpose(enc_x_lens[s:t,:])
            batch_enc_t = np.transpose(enc_t[s:t,:,:], [1, 2, 0])
            #enc_t[s:t,:]
            batch_enc_t_lens = np.transpose(enc_t_lens[s:t,:])
            #enc_t_lens[s:t]
            batch_dec_y = dec_y[s:t,:]
            batch_dec_y_lens = dec_y_lens[s:t]
            batch_select = select[s:t,:]
            batch_select_topics = select_topics[s:t,:]
            batch_select_topics_len = select_topics_len[s:t]
            

            feed_dict = {self.context: batch_enc_x,
                            self.turn_num: batch_dialog_lens,
                            self.context_lens: batch_enc_x_lens,
                            self.topic: batch_enc_t,
                            self.topic_len: batch_enc_t_lens,
                            self.response: batch_dec_y,
                            self.response_len: batch_dec_y_lens,
                            self.select: batch_select,
                            self.select_topics: batch_select_topics,
                            self.select_topics_len: batch_select_topics_len}
            predicted, select_word= self.session.run([self.predicting_ids,self.select_word], feed_dict = feed_dict)
            #loss_values.append(loss_val)
            for i in range(opts.batch_size):
                c = []
                for line in batch_enc_x[:,:,i]:                    
                    str_line = ' '.join([opts.reverse_vocabulary[n] for n in line if n not in[0,1,2]])
                    if len(str_line)>1 or str_line not in ['',' ']:
                        c.append(str_line)
                t = []
                for line in batch_enc_t[:,:,i]:                    
                    str_line = ' '.join([opts.reverse_vocabulary[n] for n in line if n not in[0,1,2]])
                    if len(str_line)>1 or str_line not in ['',' ']:
                        t.append(str_line)
                context = ' || '.join(c)
                topic = ' || '.join(t)
                select_text = ' '.join([opts.reverse_vocabulary[n] for n in select_word[i] if n not in[0,1,2]])                
                #topic = ' '.join([opts.reverse_vocabulary[n] for n in batch_enc_t[i] if n not in[0,1,2]])
                ref = ' '.join([opts.reverse_vocabulary[n] for n in batch_dec_y[i] if n not in[0,1,2]])
                
                flag = True
                pre = []
                for n in predicted[i]: 
                    if flag:
                        if n==2:
                            flag = False
                        else:
                            if n not in [0, 1]:
                                pre.append(opts.reverse_vocabulary[n])
                    else:
                        pass      
                
                
                pre_sen = ' '.join(pre)
                
                #pre = ' '.join([opts.reverse_vocabulary[n] for n in predicted[i] if n not in[0,1,2]])

                ref_lines.append(ref)
                pre_lines.append(pre_sen)
                sample = dict()
                sample['Context']=context
                sample['Topic']=topic
                sample['Select Topic'] = select_text
                sample['Target'] = ref
                sample['Predict'] = pre_sen
                test_samples.append(sample)
        avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4, dis_1, dis_2, average, extrema, greedy = self.eval(pre_lines,ref_lines)
        #print('TEST Loss: {}'.format(sum(loss_values)/len(loss_values)))
        print('TEST BLEU: {}, {}, {}, {}'.format(avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4))
        print('TEST DISTINCT: {}, {}'.format(dis_1, dis_2))   
        print('TEST Embedding: {}, {}, {}'.format(average, extrema, greedy))
            
        
        return test_samples


