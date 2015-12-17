"""
Main trainer function
"""
import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

import homogeneous_data as hd

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from training_utils import *
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
from optim import adam
from training_model import init_params, build_model
from vocab import load_dictionary

# main trainer
def trainer(Xs,
            Xs_val,
            dim_word=620, # word vector dimensionality
            dim=2400, # the number of GRU units
            encoder='gru',
            decoder='gru',
            max_epochs=5,
            dispFreq=1,
            decay_c=0.,
            grad_clip=5.,
            n_words=20000,
            maxlen_w=30,
            optimizer='adam',
            batch_size = 64,
            saveto='/u/rkiros/research/semhash/models/toy.npz',
            dictionary='/ais/gobi3/u/rkiros/bookgen/book_dictionary_large.pkl',
            embeddings=None,
            saveFreq=1000,
            reload_=False):

    # Model options
    model_options = {}
    model_options['dim_word'] = dim_word
    model_options['dim'] = dim
    model_options['encoder'] = encoder
    model_options['decoder'] = decoder 
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['decay_c'] = decay_c
    model_options['grad_clip'] = grad_clip
    model_options['n_words'] = n_words
    model_options['maxlen_w'] = maxlen_w
    model_options['optimizer'] = optimizer
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['dictionary'] = dictionary
    model_options['embeddings'] = embeddings
    model_options['saveFreq'] = saveFreq
    model_options['reload_'] = reload_

    print model_options

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'reloading...' + saveto
        with open('%s.pkl'%saveto, 'rb') as f:
            models_options = pkl.load(f)

    # load dictionary
    print 'Loading dictionary...'
    worddict = load_dictionary(dictionary)

    # Load pre-trained embeddings, if applicable
    if embeddings:
        print 'Loading embeddings...'
        from gensim.models import Word2Vec as word2vec
        embed_map = word2vec.load_word2vec_format(embeddings, binary=True)
        model_options['dim_word'] = dim_word = embed_map.vector_size
        preemb = norm_weight(n_words, dim_word)
        preemb_mask = numpy.ones((n_words, 1), dtype='float32')
        for w,i in worddict.items()[:n_words-2]:
            if w in embed_map:
                preemb[i] = embed_map[w]
                preemb_mask[i] = 0 # don't propagate gradients into pretrained embs
    else:
        preemb = None

    # Inverse dictionary
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    print 'Building model'
    params = init_params(model_options, preemb=preemb)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, x, x_mask, y, y_mask, z, z_mask, \
          opt_ret, \
          cost = \
          build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask, z, z_mask]

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=False)
    print 'Done'

    # weight decay, if applicable
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=False)
    print 'Done'

    print 'Done'
    print 'Building f_grad...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
    f_weight_norm = theano.function([], [(t**2).sum() for k,t in tparams.iteritems()], profile=False)

    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (grad_clip**2),
                                           g / tensor.sqrt(g2) * grad_clip,
                                           g))
        grads = new_grads

    if embeddings:
        param_preemb_mask = theano.shared(preemb_mask, name='preemb_mask', broadcastable=(False, True))
        grads[0] *= param_preemb_mask

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)

    print 'Optimization'

    # Each sentence in the minibatch have same length (for encoder)
    if type(Xs[0]) is not list: Xs = [Xs]
    if type(Xs_val[0]) is not list: Xs_val = [Xs_val]
    trainXs = map(hd.grouper, Xs)
    valXs = map(hd.grouper, Xs_val)
    train_iters = [hd.HomogeneousData(trainX, batch_size=batch_size, maxlen=maxlen_w) for trainX in trainXs]
    val_iters = [hd.HomogeneousData(valX, batch_size=batch_size, maxlen=maxlen_w) for valX in valXs]

    f_progress = open('%s_progress.txt' % saveto, 'w', 1)
    uidx = 0
    lrate = 0.01
    for eidx in xrange(max_epochs):
        n_samples = 0

        print 'Epoch ', eidx

        for train_iter in train_iters:
            for x, y, z in train_iter:
                n_samples += len(x)
                uidx += 1

                x, x_mask, y, y_mask, z, z_mask = hd.prepare_data(x, y, z, worddict, maxlen=maxlen_w, n_words=n_words)

                if x == None:
                    print 'Minibatch with zero sample under length ', maxlen_w
                    uidx -= 1
                    continue

                ud_start = time.time()
                cost = f_grad_shared(x, x_mask, y, y_mask, z, z_mask)
                f_update(lrate)
                ud = time.time() - ud_start

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

                if numpy.mod(uidx, saveFreq) == 0:
                    val_logprob = n_val_samples = 0
                    for val_iter in val_iters:
                        for x, y, z in val_iter:
                            n_val_samples += len(x)
                            x, x_mask, y, y_mask, z, z_mask = hd.prepare_data(x, y, z, worddict, maxlen=maxlen_w, n_words=n_words)
                            val_logprob += f_log_probs(x, x_mask, y, y_mask, z, z_mask)
                    val_logprob /= n_val_samples
                    print 'LOGPROB: %s' % val_logprob
                    f_progress.write('%s\n' % val_logprob)

                    print 'Saving...',
                    params = unzip(tparams)
                    numpy.savez('%s_%.3f' % (saveto, val_logprob), history_errs=[], **params)
                    pkl.dump(model_options, open('%s_%.3f.pkl'%(saveto, val_logprob), 'wb'))
                    print 'Done'

            print 'Seen %d samples'%n_samples

if __name__ == '__main__':
    pass


