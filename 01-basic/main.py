import tensorflow as tf


def simple_math():
    # Lets do simple python first
    a = 1
    b = 2
    c = a + b
    print("c = a(1) + b(2) =", c)

    # Lets try with tensorflow variables now
    a = tf.constant(1)
    b = tf.constant(2)
    c = a + b
    print("c = a(1) + b(2) =", c)

    # In order to get a value out of tensorflow, we must "run" the tenors in a
    # session
    with tf.Session() as session:
        result = session.run(c)
        print("c = a(1) + b(2) =", result)


def minimize():
    # y = (x - a) * (x - b) is a "parabolic function" with its lowest point in
    # the middle of 'a' and 'b'

    # Let 'a' and 'b' be placeholders, we will assign values to them later
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    x = tf.Variable(0., dtype=tf.float32)
    y = (x - a) * (x - b)

    train_op = tf.train.AdamOptimizer(0.01).minimize(y)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # Lets try for a = 0, b = 4
        print("trying to find the minimum value of 'y' for "
              "'y = (x - 0) * (x - 4)'")
        placehoders = {a: 0, b: 4}
        for _ in range(30):
            for _ in range(20):
                session.run(train_op, placehoders)
            print(session.run((x, y), placehoders))

        # Now lets try for a = 4, b = 8
        print("trying to find the minimum value of 'y' for "
              "'y = (x - 4) * (x - 8)'")
        placehoders = {a: 4, b: 8}
        for _ in range(30):
            for _ in range(20):
                session.run(train_op, placehoders)
            print(session.run((x, y), placehoders))


def main():
    simple_math()
    minimize()


if __name__ == "__main__":
    main()
