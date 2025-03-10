import os


def get_training_files(training_dir):
    return [f.replace('.jpg', '') for f in os.listdir(training_dir)]


def delete_files_in_directory(directory, files):
    # 删除directory中以files为前缀的文件
    for file in files:
        for f in os.listdir(directory):
            if f.startswith(file):
                os.remove(os.path.join(directory, f))


def main():
    training_dir = '/home/kmh/mcts/data/ISIC2016GREY/raw/train/image'  # 替换为训练集目录的路径
    # 获取目录下所有的文件，并移除.jpg后缀打印出来

    isic2016grey_dir = '/home/kmh/mcts/result/mcts/ISIC2016GREY'  # 替换为ISIC2016GREY目录的路径

    training_files = get_training_files(training_dir)
    delete_files_in_directory(isic2016grey_dir, training_files)


if __name__ == '__main__':
    main()
