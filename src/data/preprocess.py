import os
import shutil


def delete_old():
    folder_path = current_dir = os.path.dirname(__file__)

    # 构建指向 'data/processed' 文件夹的相对路径
    processed_data_dir = os.path.join(current_dir, '../../data/processed')
    # 确保路径存在
    if os.path.exists(folder_path):
        # 遍历文件夹中的所有文件和子文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # 如果是文件夹，递归删除
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                # 如果是文件，删除文件
                else:
                    os.remove(file_path)
            except Exception as e:
                print(f"Error: {file_path} : {e}")

        print(f"All contents in '{folder_path}' have been deleted.")
    else:
        print(f"The folder '{folder_path}' does not exist.")
        
if __name__ == 'main':
    delete_old()