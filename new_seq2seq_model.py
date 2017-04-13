import numpy as np
import  tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.rnn import LSTMCell
import matplotlib.pyplot as plt
import helpers



tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1

en_vocab_size = 50
vi_vocab_size = 50
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units*2

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name="encoder_inputs")
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name="encoder_inputs_length")
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name="decoder_target")
decoder_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name="decoder_inputs_length")
en_embeddings = tf.placeholder(shape=(en_vocab_size, input_embedding_size), dtype=tf.float32, name="en_embeddings")
vi_embeddings = tf.placeholder(shape=(vi_vocab_size, input_embedding_size), dtype=tf.float32, name="vi_embeddings")

en_encoder_inputs_embeded = tf.nn.embedding_lookup(en_embeddings, encoder_inputs)

encoder_cell = LSTMCell(encoder_hidden_units)

((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=en_encoder_inputs_embeded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32,
                                    time_major=True)
    )

encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)

#decoder
decoder_cell = LSTMCell(decoder_hidden_units)
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

#output projection
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vi_vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vi_vocab_size]), dtype=tf.float32)

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(vi_embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(vi_embeddings, pad_time_slice)

def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_targets_length)
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None
    return (initial_elements_finished, initial_input, initial_cell_state,
            initial_cell_output, initial_loop_state)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(vi_embeddings, prediction)
        return next_input

    elements_finished = (time >= decoder_targets_length)
    finished = tf.reduce_all(elements_finished)
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)

    state = previous_state
    output = previous_output
    loop_state = None

    return elements_finished, input, state, output, loop_state

def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)


decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vi_vocab_size))

decoder_prediction = tf.argmax(decoder_logits, 2)

#optimizer
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vi_vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

batch_size = 64
sources = helpers.random_sequences(length_from=3, length_to=10,
                                   vocab_lower=1, vocab_upper=en_vocab_size,
                                   batch_size=batch_size)

targets = helpers.random_sequences(length_from=3, length_to=10,
                                   vocab_lower=1, vocab_upper=vi_vocab_size,
                                   batch_size=batch_size)

# en_train = "data/train.ids.en"
# vi_train = "data/train.ids.vi"
# batches = helpers.read_data(en_train, vi_train, batch_size)

en_embeddings_inputs = np.random.random([en_vocab_size, input_embedding_size])
vi_embeddings_inputs = np.random.random([vi_vocab_size, input_embedding_size])

def next_feed():
    source, target = next(sources), next(targets)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(source)
    decoder_targets_, decoder_targets_length_ = helpers.batch(
        [(sequence) + [EOS] + [PAD] * 2 for sequence in target]
    )
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
        decoder_targets_length: decoder_targets_length_,
        en_embeddings: en_embeddings_inputs,
        vi_embeddings: vi_embeddings_inputs,
    }

loss_track = []

max_batches = 30001
batches_in_epoch = 1000

for batch in range(max_batches):
    fd = next_feed()
    _, l = sess.run([train_op, loss], fd)
    loss_track.append(l)

    if batch % batches_in_epoch == 0:
        print('batch {}'.format(batch))
        print('  minibatch loss: {}'.format(sess.run(loss, fd)))
        predict_ = sess.run(decoder_prediction, fd)
        for i, (dec, pred) in enumerate(zip(fd[decoder_targets].T, predict_.T)):
            print('  sample {}:'.format(i + 1))
            print('    target    > {}'.format(dec))
            print('    predicted > {}'.format(pred))
            if i >= 2:
                break
        print()

plt.plot(loss_track)
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))
plt.show()
