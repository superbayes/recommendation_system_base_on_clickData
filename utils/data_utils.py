import pandas as pd
import numpy as np
import os

def create_dir_path(path):
    """
    创建文件夹目录
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
        print('成功创建路径:{}'.format(path))


def generate_matrix(
        rating_path='../data/ratingsProcessed.csv',
        moviesinfo_path='../data/moviesProcessed.csv',
        rating_save_path='../data/rating.npy',
        record_save_path='../data/record.npy'
                    ):
    """
    ratingsProcessed的字段:userId,movieRow,rating

    moviesProcessed的字段:movieRow,movieId,title
    """

    # 读入csv标准数据
    ratings_df = pd.read_csv(rating_path)
    movies_df = pd.read_csv(moviesinfo_path)

    # 获取总用户数,产品个数 ,初始化一个(产品数,用户数)的矩阵
    userNo = ratings_df['userId'].max() + 1
    productNo = ratings_df['movieRow'].max() + 1
    rating = np.zeros((productNo, userNo))

    # 将所有用户对所有电影的评分数据转变为一个矩阵rating
    flag = 0  # 记录处理进度
    ratings_df_length = np.shape(ratings_df)[0]  # 100836行数据
    for index, row in ratings_df.iterrows():  # 获取ratings_df的每一行
        rating[int(row['movieRow']), int(row['userId'])] = row['rating']
        flag += 1  # 表示处理完一行
        if flag % 50000 == 0:
            print('processed %d,%d left' % (flag, ratings_df_length - flag))

    # 生成一个产品>>用户矩阵record,若用户对产品有评分则为1无评分则为0
    record = rating > 0
    record = np.array(record, dtype=int)

    # 将用户对产品的评分矩阵rating缩放到0-1,并保存rating,record矩阵
    rating = rating / 5

    # 将矩阵进行处理,是nan值转变为0
    rating = np.nan_to_num(rating)
    record = np.nan_to_num(record)


    # 保存电影评分矩阵rating和评分记录矩阵record
    np.save(record_save_path, record)
    np.save(rating_save_path, rating)

    print('保存保险产品点击评分矩阵rating和评分记录矩阵record成功!!-------------------')
    print('产品数:{},用户数{}'.format(productNo, userNo))
    print('评分矩阵rating和评分记录矩阵record的形状是: ', record.shape)
    print('评分矩阵rating和评分记录矩阵record的保存路径: ', rating_save_path)
    return None


def load_data(
        ratings_df_path='../data/ratingsProcessed.csv',
        movies_df_path='../data/moviesProcessed.csv',
        rating_path='../data/rating.npy',
        record_path='../data/record.npy'
                ):
    """

    :param ratings_df_path:
    :param movies_df_path:
    :param rating_path:
    :param record_path:
    :return:
    """
    record = np.load(record_path)
    rating = np.load(rating_path)
    ratings_df = pd.read_csv(ratings_df_path)
    movies_df = pd.read_csv(movies_df_path)

    # m:产品数, n:用户数
    m, n = rating.shape

    return ratings_df, movies_df, rating, record, m, n

if __name__ == '__main__':
    generate_matrix()