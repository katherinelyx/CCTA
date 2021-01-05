import pickle
import argparse
import numpy as np
#from IFTC import Options, IFTC
from CCTA import Options, HT
#import read_ht
import tensorflow as tf
import tqdm
from gensim.models import Word2Vec, KeyedVectors
import math


# Parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type = str, default = 'data/DD/train',
                    help = 'the directory to the train data')
parser.add_argument('--test_data_path', type = str, default = 'data/DD/test',
                    help = 'the directory to the test data')
parser.add_argument('--dev_data_path', type = str, default = 'data/DD/test',
                    help = 'the directory to the dev data')                    
parser.add_argument('--num_epochs', type = int, default = 20,
                    help = 'the number of epochs to train the data')
parser.add_argument('--batch_size', type = int, default = 64,
                    help = 'the batch size')
parser.add_argument('--learning_rate', type = float, default = 0.001,
                    help = 'the learning rate')
parser.add_argument('--beam_width', type = int, default = 0,
                    help = 'the beam width when decoding')
parser.add_argument('--embedding_size', type = int, default = 512,
                    help = 'the size of word embeddings')
parser.add_argument('--embedding_path', type = str, default = 'GoogleNews-vectors-negative300.bin',#utils/sgns.weibo.word   GoogleNews-vectors-negative300.bin
                    help = 'the path of word embeddings file')
parser.add_argument('--num_hidden_units', type = int, default = 512,
                    help = 'the number of hidden units')
parser.add_argument('--save_path', type = str, default = 'model/',
                    help = 'the path to save the trained model to')
parser.add_argument('--restore_path', type = str, default = 'model/',
                    help = 'the path to restore the trained model')
parser.add_argument('--restore', type = bool, default = False,
                    help = 'whether to restore from a trained model')
parser.add_argument('--predict', type = bool, default = True,
                    help = 'whether to enter predicting mode')
parser.add_argument('--training', type = bool, default = False,
                    help = 'whether to enter training mode')
args = parser.parse_args()

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
    with open('{}/vocabulary.pickle'.format(data_path), 'rb') as file:
        vocabulary = pickle.load(file)
    with open('{}/vocabulary_reverse.pickle'.format(data_path), 'rb') as file:
        vocabulary_reverse = pickle.load(file)
    
    return enc_x, context_lens, enc_x_lens, enc_t, enc_t_lens, dec_y, dec_y_lens, select, select_topics, select_topics_len, vocabulary, vocabulary_reverse

def load_wordvector(embeddings_file, language="English"):
    print("loading embeddings file...")
    if language == 'English':
        w2v = KeyedVectors.load_word2vec_format(embeddings_file, binary=True)
    else:
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            words = set()
            w2v = {}
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                w2v[curr_word] = np.array(line[1:], dtype=np.float32)
    return w2v



if __name__ == '__main__':
    
    
    print('Load Data----------------')
    enc_x, context_lens, enc_x_lens, enc_t, enc_t_lens, dec_y, dec_y_lens, select, select_topics, select_topics_len, vocabulary, vocabulary_reverse = read_data(args.train_data_path)
    max_dialog_len = enc_x.shape[1]
    max_utterance_len = enc_x.shape[2]
    max_topic_len = enc_t.shape[2]
    label_len = select.shape[1]
    
    w2v = []

    print('Set parametres------------------')
    options = Options(reverse_vocabulary = vocabulary_reverse,num_epochs = args.num_epochs,
                      batch_size = args.batch_size,
                      learning_rate = 0.001,
                      lr_decay = 0.995,
                      max_grad_norm = 5.0,
                      beam_width = args.beam_width,
                      vocabulary_size = len(vocabulary),                      
                      embedding_size = args.embedding_size,
                      num_hidden_layers = args.num_hidden_layers,
                      num_hidden_units = args.num_hidden_units,
                      max_dialog_len = max_dialog_len,
                      max_utterance_len = max_utterance_len,
                      max_topic_len = max_topic_len,
                      label_len = label_len,
                      go_index = vocabulary['<go>'],
                      eos_index = vocabulary['<eos>'],
                      training_mode = args.training,
                      num_blocks = 6,
                      num_heads = 8,
                      save_model_path = args.save_path,
                      dev_data_path = args.dev_data_path,
                      embedding_file = w2v)
    
       

    model = HT(options)

    if args.predict:
        model.restore(args.restore_path)
        # predicted : list, element is dict like ['Target':. 'Predict':.]
        predicted= model.predict(args.test_data_path)
                    
        with open('result/CCTA.txt', 'w', encoding= 'utf-8') as f:
            for item in predicted: 
                f.write('Context: ')
                f.write(item['Context'])
                f.write('\n')
                f.write('Context Topic: ')
                f.write(item['Topic'])
                f.write('\n')
                f.write('Select Topic: ')
                f.write(item['Select Topic'])
                f.write('\n')              
                f.write('Target: ')
                f.write(item['Target'])
                f.write('\n')
                f.write('Predict: ')
                f.write(item['Predict'])
                f.write('\n\n')
        
    else:
        if args.restore:
            model.restore(args.restore_path)
        else:
            model.init_tf_vars()
        model.train(enc_x, dec_y, enc_t, context_lens, enc_x_lens, dec_y_lens, enc_t_lens, select, select_topics, select_topics_len)
        #model.save(args.save_path)
