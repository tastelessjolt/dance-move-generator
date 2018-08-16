import tensorflow as tf
import numpy as np
import wave
import re

BATCH_SIZE = 12
NUM_CLASSES = 49
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 32
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 40

MOVING_AVERAGE_DECAY = 0.999        # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0        # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1    # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1         # Initial learning rate.


TOWER_NAME = 'tower'

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _generate_audio_and_label_batch(audio, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of audios and labels.

  Args:
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of audios per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    audios: audios. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  # Create a queue that shuffles the examples, and then
  # read 'batch_size' audios + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    audios, label_batch = tf.train.shuffle_batch(
        [audio, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        enqueue_many=True,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    audios, label_batch = tf.train.batch(
        [audio, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        enqueue_many=True,
        capacity=min_queue_examples + 3 * batch_size)

  # print(label_batch)
  # Display the training audios in the visualizer.
  # tf.summary.audio('audios', audios)

  return audios, tf.reshape(label_batch, [batch_size])
  # return audios, label_batch

def distorted_inputs(song_name):
    file = open('all.dat', 'r')
    all_labels = list(file)
    all_labels = [x.strip('\n') for x in all_labels]
    NUM_CLASSES = len(all_labels)
    file.close()

    dictionary = {}
    for i in range(len(all_labels)):
        dictionary[all_labels[i]] = i
    # print(dictionary)

    file = open('training/' + song_name +  '.dat', 'r')
    label = list(file)
    ntraining = len(label)
    label = [x.strip('\n') for x in label]
    label = [dictionary[x] for x in label]
    # print(np.ndarray(label).T)
    label = tf.convert_to_tensor(label)
    file.close()

    file = wave.open('training/' + song_name + '.wav', 'rb')
    s = file.readframes(file.getnframes())
    nparr = np.array([float(x) for x in list(s)])
    record_bytes = tf.convert_to_tensor(list((nparr)/(nparr.max())))
    fin = tf.slice(record_bytes, [0] ,[ ((file.getnframes() * 4) - ((file.getnframes() * 4) % ntraining))])
    # fin = tf.expand_dims(fin, 1)
    print(fin.shape)
    # print(fin.shape)
    fin = tf.random_crop(record_bytes, [ntraining * 44100 * 4])
    audio = tf.reshape(fin, [(ntraining) , 44100 * 4, 1])
    file.close()


    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    return _generate_audio_and_label_batch(audio, label, min_queue_examples, BATCH_SIZE,shuffle=False)

def inputs(song_name):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    audios: audios. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  file = open('all.dat', 'r')
  all_labels = list(file)
  all_labels = [x.strip('\n') for x in all_labels]
  NUM_CLASSES = len(all_labels)
  file.close()

  dictionary = {}
  for i in range(len(all_labels)):
      dictionary[all_labels[i]] = i
  # print(dictionary)

  file = open('testing/' + song_name +  '.dat', 'r')
  label = list(file)
  ntraining = len(label)
  label = [x.strip('\n') for x in label]
  label = [dictionary[x] for x in label]
  # print(np.ndarray(label).T)
  label = tf.convert_to_tensor(label[1:])
  file.close()

  file = wave.open('testing/' + song_name + '.wav', 'rb')
  ntraining = int(file.getnframes() / 44100);
  s = file.readframes(file.getnframes())
  nparr = np.array([float(x) for x in list(s)])
  record_bytes = tf.convert_to_tensor(list((nparr)/(nparr.max())))
  # record_bytes = tf.convert_to_tensor(list((nparr - nparr.mean())/(nparr.std())))
  # record_bytes = tf.cast(tf.decode_raw(tf.convert_to_tensor(s), tf.uint8), tf.float32)
  fin = tf.random_crop(record_bytes, [(ntraining) * 44100 * 4])
  audio = tf.reshape(fin, [(ntraining ) , 44100 * 4, 1])
  file.close()

  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # inference(audio)
  return _generate_audio_and_label_batch(audio, label, min_queue_examples, BATCH_SIZE, shuffle=False)
  # return audio

def inference(audio):
    #conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                            shape=[200, 1, 1],
                                            stddev=5e-2,
                                            wd=0.0)
        # print(audio)
        conv = tf.nn.conv1d(audio, kernel, 1, padding="SAME")
        biases = _variable_on_cpu('biases', [1], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # list(conv1.shape)
    # print(tf.expand_dims(conv1, 1))
    pool1 = tf.nn.max_pool(tf.expand_dims(conv1, 1), ksize=[1, 1, 200, 1],
                        strides=[1, 1, 3, 1], padding='SAME', name='pool1')

    #norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm1')

    print(norm1.shape)

    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm1, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                        stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    #linear layer(WX + b)
    with tf.variable_scope('softmax_linear') as scope:

        weights = _variable_with_weight_decay('weights', [384, NUM_CLASSES],
                                                stddev=1/384.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local3, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
  """Add summaries for losses in dance model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
    # Variable that affect learning rate
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving avarages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    return train_op
