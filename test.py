from sparse import *


save_path = os.path.join(args.save_dir + 'model_LR=0.005')
file_names = os.listdir(args.gaze_path)
y_, maxi, mini = normalize_input(csv_loader(file_names), 'Normalization')
d = tf.get_variable('D', shape=[np.shape(y_)[0], args.dimension2])
x = tf.get_variable('X', shape=[args.dimension2, np.shape(y_)[1]])
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_path)
    print(' Model Restore Successful!')
    d = de_normalized(d.eval(), maxi, mini)
    print('D: ', d)
    x = de_normalized(x.eval(), maxi, mini)
    print('X: ', x)
