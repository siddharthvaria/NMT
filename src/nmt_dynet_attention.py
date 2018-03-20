from dynet import *
import argparse
from utils import Corpus
import random
import numpy as np
from bleu import get_bleu_score
import json

RNN_BUILDER = GRUBuilder

class nmt_dynet_attention:

    def __init__(self, src_vocab_size, tgt_vocab_size, src_word2idx, src_idx2word, tgt_word2idx, tgt_idx2word, word_d, gru_d, gru_layers):

        # initialize variables
        self.gru_layers = gru_layers
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_word2idx = src_word2idx
        self.src_idx2word = src_idx2word
        self.tgt_word2idx = tgt_word2idx
        self.tgt_idx2word = tgt_idx2word
        self.word_d = word_d
        self.gru_d = gru_d

        self.model = Model()

        # the embedding paramaters
        self.source_embeddings = self.model.add_lookup_parameters((self.src_vocab_size, self.word_d))
        self.target_embeddings = self.model.add_lookup_parameters((self.tgt_vocab_size, self.word_d))

        # encoder networks
        # the foreword rnn
        self.fwd_RNN = RNN_BUILDER(self.gru_layers, self.word_d, self.gru_d, self.model)
        # the backword rnn
        self.bwd_RNN = RNN_BUILDER(self.gru_layers, self.word_d, self.gru_d, self.model)

        # decoder network
        self.dec_RNN = RNN_BUILDER(self.gru_layers, self.word_d + 2 * self.gru_d, self.word_d + 2 * self.gru_d, self.model)

        # project the decoder output to a vector of tgt_vocab_size length
        self.output_w = self.model.add_parameters((self.tgt_vocab_size, self.word_d + 2 * self.gru_d))
        self.output_b = self.model.add_parameters((self.tgt_vocab_size))

        # attention weights
        self.attention_w1 = self.model.add_parameters((2 * self.gru_d, 2 * self.gru_d))
        self.attention_w2 = self.model.add_parameters((2 * self.gru_d, self.word_d + 2 * self.gru_d))
        self.attention_v = self.model.add_parameters((1, 2 * self.gru_d))

    def _run_rnn1(self, init_state, input_vecs):
        s = init_state

        states = s.add_inputs(input_vecs)
        rnn_outputs = [s.output() for s in states]
        return rnn_outputs

    def _get_probs(self, rnn_output):
        output_w = parameter(self.output_w)
        output_b = parameter(self.output_b)
        probs = softmax(output_w * rnn_output + output_b)
        return probs

    def _embed_src_sentence(self, src_sentence):
#         print src_sentence
        return [self.source_embeddings[w] for w in src_sentence]

    def _embed_tgt_sentence(self, tgt_sentence):
        return [self.target_embeddings[w] for w in tgt_sentence]

    def _predict(self, probs):
        probs = probs.value()
        predicted_word = self.tgt_idx2word[probs.index(max(probs))]
        return predicted_word

    def _encode_string(self, src_sentence):

        embedded_src_sentence = self._embed_src_sentence(src_sentence)

        # run the foreword RNN
        rnn_fwd_state = self.fwd_RNN.initial_state()
        rnn_fwd_outputs = self._run_rnn1(rnn_fwd_state, embedded_src_sentence)

        # run the backword rnn
        rnn_bwd_state = self.bwd_RNN.initial_state()
        rnn_bwd_outputs = self._run_rnn1(rnn_bwd_state, embedded_src_sentence[::-1])[::-1]

        # concataenate the foreward and backword outputs
        rnn_outputs = [concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(rnn_fwd_outputs, rnn_bwd_outputs)]

        return rnn_outputs

    def _attend(self, input_vectors, state):

        '''
        input_vectors: hidden states of the encoder
        state: previous state of the decoder
        '''

        attention_w1 = parameter(self.attention_w1)
        attention_w2 = parameter(self.attention_w2)
        attention_v = parameter(self.attention_v)

        attention_weights = []

        w2dt = attention_w2 * state.h()[-1]
        for input_vector in input_vectors:
            attention_weight = attention_v * tanh(attention_w1 * input_vector + w2dt)
            attention_weights.append(attention_weight)
        attention_weights = softmax(concatenate(attention_weights))

        output_vectors = esum(
            [vector * attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])
        return output_vectors

    def get_loss(self, src_sentence, tgt_sentence):

        # src_sentence: words in src sentence
        # tgt_sentence: words in tgt sentence
        renew_cg()

        src_sentence = [self.src_word2idx[w] if w in self.src_word2idx else self.src_word2idx['<unk>'] for w in src_sentence]
        # convert sentence of indices to sentence of word embeddings

        tgt_sentence = [self.tgt_word2idx[w] for w in tgt_sentence]
        # embedded_tgt_sentence = self._embed_tgt_sentence(tgt_sentence)

        # The encoded string is the hidden state of the last slice of the encoder
        # encoded_string = self._encode_string(embedded_string)[-1]
        encoded_string = self._encode_string(src_sentence)

        # rnn_state = self.dec_RNN.initial_state([encoded_string])

        rnn_state = self.dec_RNN.initial_state().add_input(vecInput(self.word_d + 2 * self.gru_d))
        # rnn_state = self.dec_RNN.initial_state()

        loss = []
        prev_word_emb = vecInput(self.word_d)
        prev_word_emb.set(np.zeros(self.word_d))

        for i in xrange(len(tgt_sentence)):
            w_id = tgt_sentence[i]
            attended_encoding = self._attend(encoded_string, rnn_state)
            rnn_state = rnn_state.add_input(concatenate([prev_word_emb, attended_encoding]))
            probs = self._get_probs(rnn_state.output())
            loss.append(-log(pick(probs, w_id)))
            prev_word_emb = self.target_embeddings[w_id]

        loss = esum(loss)
        return loss

    def generate(self, src_sentence):
        renew_cg()

        src_sentence = [self.src_word2idx[w] if w in self.src_word2idx else self.src_word2idx['<unk>'] for w in src_sentence]

        encoded_string = self._encode_string(src_sentence)

        # rnn_state = self.dec_RNN.initial_state([encoded_string])
        rnn_state = self.dec_RNN.initial_state().add_input(vecInput(self.word_d + 2 * self.gru_d))
        # rnn_state = self.dec_RNN.initial_state()

        tgt_sentence = []
        prev_word_emb = vecInput(self.word_d)
        prev_word_emb.set(np.zeros(self.word_d))

        while True:
            attended_encoding = self._attend(encoded_string, rnn_state)
            rnn_state = rnn_state.add_input(concatenate([prev_word_emb, attended_encoding]))
            probs = self._get_probs(rnn_state.output())
            predicted_word = self._predict(probs)
            tgt_sentence.append(predicted_word)
            prev_word_emb = self.target_embeddings[self.tgt_word2idx[predicted_word]]
            if predicted_word == '</s>' or len(tgt_sentence) > 2 * len(src_sentence):
                break
#         output_string = ' '.join(output_string)
        return tgt_sentence

    def translate_all(self, src_sentences):
        translated_sentences = []
        for src_sentence in src_sentences:
            # print src_sentence
            translated_sentences.append(self.generate(src_sentence))

        return translated_sentences

    # save the model, and optionally the word embeddings
    def save(self, filename):

        self.model.save(filename)
        embs = {}
        if self.src_idx2word:
            src_embs = {}
            for i in range(self.src_vocab_size):
                src_embs[self.src_idx2word[i]] = self.source_embeddings[i].value()
            embs['src'] = src_embs

        if self.tgt_idx2word:
            tgt_embs = {}
            for i in range(self.tgt_vocab_size):
                tgt_embs[self.tgt_idx2word[i]] = self.target_embeddings[i].value()
            embs['tgt'] = tgt_embs

        if len(embs):
            with open(filename + '_embeddings.json', 'w') as f:
                json.dump(embs, f)

def get_val_set_loss(network, val_set):
        loss = []
        for src_sentence, tgt_sentence in zip(val_set.source_sentences, val_set.target_sentences):
            loss.append(network.get_loss(src_sentence, tgt_sentence).value())

        return sum(loss)

def main(train_src_file, train_tgt_file, dev_src_file, dev_tgt_file, model_file, num_epochs, embeddings_init = None, seed = 0):
    print('reading train corpus ...')
    train_set = Corpus(train_src_file, train_tgt_file)
    # assert()
    print('reading dev corpus ...')
    dev_set = Corpus(dev_src_file, dev_tgt_file)

    # test_set = Corpus(test_src_file)

    print 'Initializing neural machine translator with attention:'
    # src_vocab_size, tgt_vocab_size, tgt_idx2word, word_d, gru_d, gru_layers
    encoder_decoder = nmt_dynet_attention(len(train_set.source_word2idx), len(train_set.target_word2idx), train_set.source_word2idx, train_set.source_idx2word, train_set.target_word2idx, train_set.target_idx2word, 50, 50, 2)

    trainer = SimpleSGDTrainer(encoder_decoder.model)

    sample_output = np.random.choice(len(dev_set.target_sentences), 5, False)
    losses = []
    best_bleu_score = 0
    for epoch in range(num_epochs):
        print 'Starting epoch', epoch
        # shuffle the training data
        combined = list(zip(train_set.source_sentences, train_set.target_sentences))
        random.shuffle(combined)
        train_set.source_sentences[:], train_set.target_sentences[:] = zip(*combined)

        print 'Training . . .'
        sentences_processed = 0
        for src_sentence, tgt_sentence in zip(train_set.source_sentences, train_set.target_sentences):
            loss = encoder_decoder.get_loss(src_sentence, tgt_sentence)
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            sentences_processed += 1
            if sentences_processed % 2000 == 0:
                print 'sentences processed: ', sentences_processed

        # Accumulate average losses over training to plot
        val_loss = get_val_set_loss(encoder_decoder, dev_set)
        print 'Validation loss this epoch', val_loss
        losses.append(val_loss)

        print 'Translating . . .'
        translated_sentences = encoder_decoder.translate_all(dev_set.source_sentences)

        print('translating {} source sentences...'.format(len(sample_output)))
        for sample in sample_output:
            print('Target: {}\nTranslation: {}\n'.format(' '.join(dev_set.target_sentences[sample]),
                                                                         ' '.join(translated_sentences[sample])))

        bleu_score = get_bleu_score(translated_sentences, dev_set.target_sentences)
        print 'bleu score: ', bleu_score
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            # save the model
            encoder_decoder.save(model_file)

    print 'best bleu score: ', best_bleu_score

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
#     parser.add_argument('model_type')
    parser.add_argument('train_src_file')
    parser.add_argument('train_tgt_file')
    parser.add_argument('dev_src_file')
    parser.add_argument('dev_tgt_file')
    parser.add_argument('model_file')
    parser.add_argument('--num_epochs', default = 20, type = int)
    parser.add_argument('--embeddings_init')
    parser.add_argument('--seed', default = 0, type = int)
    parser.add_argument('--dynet-mem')

    args = vars(parser.parse_args())
    args.pop('dynet_mem')

    main(**args)
