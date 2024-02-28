import pickle
import tensorflow as tf


TRAIN_BUF = 60000
VAL_BUF = 1000
TEST_BUF = 9000


def create_numbers(N, data_x, data_y, batch_size=10, log=True):
    idx_perm = tf.random.shuffle(tf.range(data_x.shape[0]))

    generator = []
    safety_length = data_x.shape[0] // (2 * N * batch_size) * (2 * N * batch_size)
    for i in range(safety_length // (2 * N)):
        sum_image1 = []
        value1 = 0
        for j in range(N):
            sum_image1.insert(0, tf.expand_dims(data_x[idx_perm[2 * N * i + j]], axis=-1))
            value1 += data_y[idx_perm[2 * N * i + j]] * 10 ** j

        sum_image2 = []
        value2 = 0
        for j in range(N, 2 * N):
            sum_image2.insert(0, tf.expand_dims(data_x[idx_perm[2 * N * i + j]], axis=-1))
            value2 += data_y[idx_perm[2 * N * i + j]] * 10 ** (j - N)

        """ Target is 0., as we optimise the log probability """
        if log:
            target = 0.
        else:
            target = 1.
        generator.append((sum_image1, sum_image2, value1 + value2, target))
    return generator


def create_loader(N, BATCH_SIZE=10, log=False):
    TRAIN_BUF = 60000 // (2 * N * BATCH_SIZE) * (BATCH_SIZE)
    VAL_BUF = 1000 // (2 * N * BATCH_SIZE) * (BATCH_SIZE)
    TEST_BUF = 9000 // (2 * N * BATCH_SIZE) * (BATCH_SIZE)

    if log:
        target = "log_prob"
    else:
        target = "prob"

    try:
        train_gen = pickle.load(open(f'examples/ADDITION/data/data_{N}_train_batch{BATCH_SIZE}_{target}.pkl', 'rb'))
        test_gen = pickle.load(open(f'examples/ADDITION/data/data_{N}_test_batch{BATCH_SIZE}_{target}.pkl', 'rb'))

        train_dataset = tf.data.Dataset.from_generator(lambda: train_gen, (tf.float32, tf.float32, tf.int64, tf.float32)).shuffle(
            TRAIN_BUF).batch(BATCH_SIZE)
        val_dataset = tf.data.Dataset.from_generator(lambda: test_gen[:VAL_BUF], (tf.float32, tf.float32, tf.int64, tf.float32)).shuffle(VAL_BUF).batch(
            BATCH_SIZE)
        test_dataset = tf.data.Dataset.from_generator(lambda: test_gen[VAL_BUF:], (tf.float32, tf.float32, tf.int64, tf.float32)).shuffle(TEST_BUF).batch(
            BATCH_SIZE)

        return train_dataset, val_dataset, test_dataset
    except FileNotFoundError:
        pass

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.  # [60000, 28, 28]
    x_test = x_test.astype('float32') / 255.  # [10000, 28, 28]


    train_gen = create_numbers(N, x_train, y_train, log=log)
    test_gen = create_numbers(N, x_test, y_test, log=log)

    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen, (tf.float32, tf.float32, tf.int64, tf.float32)).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_generator(lambda: test_gen[:VAL_BUF], (tf.float32, tf.float32, tf.int64, tf.float32)).shuffle(VAL_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_generator(lambda: test_gen[VAL_BUF:], (tf.float32, tf.float32, tf.int64, tf.float32)).shuffle(TEST_BUF).batch(BATCH_SIZE)

    pickle.dump(train_gen, open(f'examples/ADDITION/data/data_{N}_train_batch{BATCH_SIZE}_{target}.pkl', 'wb'))
    pickle.dump(test_gen, open(f'examples/ADDITION/data/data_{N}_test_batch{BATCH_SIZE}_{target}.pkl', 'wb'))

    return train_dataset, val_dataset, test_dataset