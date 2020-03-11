import tensorflow as tf
import numpy as np
import os
from utils.data_utils import *



def train(model_save_path = './models/checkpoints',  # 模型保存路径
          log_path='./models/logPath',  # 日志保存路径
          num_features = 10,  # 隐变量特征个数
          epoches=100,
          is_train=True,  # 采取True模式则不断更新矩阵参数, 否则不更新
          lr=1e-4  # 学习率

          ):

    # 生成模型需要的数据,并且加载数据
    generate_matrix(rating_path='./data/ratingsProcessed.csv',
                    moviesinfo_path='./data/moviesProcessed.csv',
                    rating_save_path='./data/rating.npy',
                    record_save_path='./data/record.npy')
    ratings_df, movies_df, rating, record, productNum, userNum = load_data( ratings_df_path='./data/ratingsProcessed.csv',
                                                                            movies_df_path='./data/moviesProcessed.csv',
                                                                            rating_path='./data/rating.npy',
                                                                            record_path='./data/record.npy'
                                                                            )

    # 初始化内容矩阵和用户喜好矩阵，产生的参数都是随机数并且是正态分布的
    X_parameters = tf.Variable(tf.random_normal([productNum, num_features], stddev=0.1, mean=0.0), trainable=is_train, name='X_parameters')
    Theta_parameters = tf.Variable(tf.random_normal([userNum, num_features], stddev=0.1, mean=0.0), trainable=is_train, name='Theta_parameters')

    # 构建损失函数 将X_parameters，Theta_parameters矩阵相乘相乘之前将Theta_parameters转置
    loss = 1/ 2 * tf.reduce_sum(
        ((tf.matmul(X_parameters, Theta_parameters, transpose_b=True) - rating) * record) ** 2) + 1 / 2 * (
                       tf.reduce_sum(X_parameters ** 2) + tf.reduce_sum(Theta_parameters ** 2))
    # fixme 可视化模型损失
    tf.summary.scalar(name='train_loss', tensor=loss, collections=['train'])

    # 构建优化器
    optimizer = tf.train.AdamOptimizer(lr)
    train = optimizer.minimize(loss)

    # 计算错误率
    predicts = tf.matmul(X_parameters, Theta_parameters, transpose_b=True)
    errors = tf.sqrt(tf.reduce_sum((predicts - rating) ** 2))
    # fixme 可视化模型损失
    tf.summary.scalar(name='train_acc', tensor=errors, collections=['train'])

    # 计算过程可视化
    train_summary = tf.summary.merge_all('train')

    # 构建持久化路径 和 持久化对象。
    saver = tf.train.Saver(max_to_keep=2)
    create_dir_path(model_save_path)

    # 二、执行会话。
    with tf.Session() as sess:
        # 0、断点继续训练(恢复模型)
        ckpt = tf.train.get_checkpoint_state(model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('加载持久化模型，断点继续训练!')
        else:
            # 1、初始化全局变量
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            print('没有持久化模型，从头开始训练!')

        # FileWriter 的构造函数中包含了参数log_dir，申明的所有事件都会写到它所指的目录下
        summary_writer = tf.summary.FileWriter(logdir=log_path, graph=sess.graph)

        # 开启训练 ################################################################################################
        step = 1

        for i in range(epoches):
            _, train_summary_, train_loss, train_errors = sess.run([train, train_summary, loss, errors])
            summary_writer.add_summary(train_summary_, global_step=step)
            print('Step:{} - 训练损失:{:.5f} - 训练错误率:{:.4f}'.format(i + 1, train_loss, train_errors))

            # 每5个批次打印一次训练精度,损失;
            if (step) % 2 == 0:
                # todo 模型持久化 每训练够5轮就保存一次模型
                file_name = '_{}_model.ckpt'.format(step)
                save_file = os.path.join(model_save_path, file_name)
                saver.save(sess=sess, save_path=save_file, global_step=step)
                print('model saved to path:{}'.format(save_file))

            # 如果错误率小于8%,则停止训练
            if train_errors <= 0.08:
                break

            step += 1  # 每训练一批,step就+1,主要是为了tf.summary和模型持久化
        # 结束训练 ################################################################################################

        summary_writer.flush()
        sess.close()
        print('训练运行成功.................................................')

def test(
        model_save_path='./models/checkpoints',
        result_matrix_save_path='./models/result_matrix.npy',
        num_features=10,
        is_train=False,
        lr=1e-4
        ):

    ratings_df, movies_df, rating, record, productNum, userNum = load_data( ratings_df_path='./data/ratingsProcessed.csv',
                                                                            movies_df_path='./data/moviesProcessed.csv',
                                                                            rating_path='./data/rating.npy',
                                                                            record_path='./data/record.npy'
                                                                            )
    #
    # 初始化内容矩阵和用户喜好矩阵，产生的参数都是随机数并且是正态分布的
    X_parameters = tf.Variable(tf.random_normal([productNum, num_features], stddev=0.1, mean=0.0), trainable=is_train, name='X_parameters')
    Theta_parameters = tf.Variable(tf.random_normal([userNum, num_features], stddev=0.1, mean=0.0), trainable=is_train, name='Theta_parameters')

    saver = tf.train.Saver(max_to_keep=2)
    create_dir_path(model_save_path)

    # 二、执行会话。
    with tf.Session() as sess:
        # 0、断点继续训练(恢复模型)
        ckpt = tf.train.get_checkpoint_state(model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('加载持久化模型，断点继续训练!')
        else:
            # 1、初始化全局变量
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            print('没有持久化模型，从头开始训练!')

        X_, Theta_ = sess.run([X_parameters, Theta_parameters])
        # 将电影内容矩阵和用户喜好矩阵相乘，再加上每一行的均值，便得到一个完整的电影评分表
        predicts = np.dot(X_, Theta_.T)
        # 保存预测矩阵(产品数,用户数)
        predicts=predicts*5
        np.save(file=result_matrix_save_path, arr=predicts)
    
    sess.close()

if __name__ == '__main__':
    train(epoches=500)
