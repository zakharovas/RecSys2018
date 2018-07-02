import tensorflow as tf
import numpy as np

from json import loads
from json import dumps

from tqdm import tqdm
from itertools import count
from functools import partial


# Model params:
user_count = 17259
item_count = 2149247

batch_size = 10000

factor_dim = 32

# Dataset initialization:

def generator(filenames):
    for filename in filenames:
        with open(filename, 'r') as source:
            for line in source:
                record = loads(line)
        
                user   = record['user']
                view   = record['view']
                weight = record['view_count']

                fill = view + [item_count] * 256
                fill = fill[:256]

                for item in view:
                    yield user, item, 1, weight, fill

                    sample = np.random.randint(0, item_count)
                    yield user, sample, 0., weight, fill


dataset = tf.data.Dataset.from_generator(
    partial(generator, ['encoded_playlists.json']),
    output_types=(tf.int64, tf.int64, tf.float64, tf.float64, tf.int64)
)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(32)

# Variables:
mean = tf.get_variable(name='mu', shape=[])

user_bias = tf.get_variable(name='b_u', shape=[user_count])
item_bias = tf.get_variable(name='b_i', shape=[item_count])

user_factor = tf.get_variable(name='p_u', shape=[user_count, factor_dim])
item_factor = tf.get_variable(name='q_i', shape=[item_count, factor_dim])

item_interaction = tf.get_variable(name='y_i', shape=[item_count, factor_dim])


# Placeholders:
users = tf.placeholder(tf.int64, shape=(None,), name='users')
items = tf.placeholder(tf.int64, shape=(None,), name='items')
rates = tf.placeholder(tf.float64, shape=(None,), name='rates')
views = tf.placeholder(tf.int64, shape=(None, None), name='views')
weights = tf.placeholder(tf.float32, shape=(None,), name='length')

# Formulae:
u = weights[:, tf.newaxis] * tf.reduce_sum(
    tf.nn.embedding_lookup(
        tf.concat([item_interaction, tf.zeros(shape=[1, factor_dim])], axis=0),
        views
    ),
    axis=1
) + tf.nn.embedding_lookup(user_factor, users)
i = tf.nn.embedding_lookup(item_factor, items)

estimate = (
    mean +
    tf.nn.embedding_lookup(item_bias, items) +
    tf.nn.embedding_lookup(user_bias, users) +
    tf.reduce_sum(
        tf.multiply(i, u),
        axis=1
    )
)

loss = tf.losses.log_loss(
    labels=rates,
    predictions=tf.sigmoid(estimate)
)

# Optimization:
optimizer = tf.train.AdamOptimizer()
training = optimizer.minimize(loss)

nx = dataset.make_one_shot_iterator().get_next()

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 32
config.inter_op_parallelism_threads = 32

with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())

    with tqdm() as widget:
        while True:
            try:
                dataline = session.run(nx)

                _, loss_value = session.run(
                    (training, loss),
                    feed_dict={
                        users : dataline[0],
                        items : dataline[1],
                        rates : dataline[2],
                        weights : dataline[3],
                        views : dataline[4]
                    }
                )
                widget.set_postfix({
                    'Loss on current user': loss_value
                })
                widget.update()
            # except tf.errors.OutOfRangeError:
            except:
                break

    user_factor = session.run(user_factor)
    user_bias = session.run(user_bias)

    with open('user_encoding.json', 'r') as source:
        with open('user_factor.json', 'w') as target:
            for line in tqdm(source):
                encoding = loads(line)
                target.write(dumps({
                    'name'  : encoding['key'],
                    'bias'  : user_bias[encoding['val']].tolist(),
                    'factor': user_factor[encoding['val'], :].tolist()
                }))
                target.write('\n')

    item_bias = session.run(item_bias)
    item_factor = session.run(item_factor)
    item_interaction = session.run(item_interaction)

    with open('item_encoding.json', 'r') as source:
        with open('item_factor.json', 'w') as target:
            for line in tqdm(source):
                encoding = loads(line)
                target.write(dumps({
                    'id'    : encoding['key'],
                    'bias'  : item_bias[encoding['val']].tolist(),
                    'factor': item_factor[encoding['val'], :].tolist(),
                    'inner' : item_interaction[encoding['val'], :].tolist()
                }))
                target.write('\n')
