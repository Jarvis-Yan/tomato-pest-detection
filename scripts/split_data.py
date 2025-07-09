import os, random, shutil


def moveimg(fileDir, tarDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    print(f"总文件数: {filenumber}")  # 打印总文件数
    rate = 0.3  # 自定义抽取图片的比例
    picknumber = int(filenumber * rate)
    print(f"抽取文件数: {picknumber}")  # 打印抽取文件数
    sample = random.sample(pathDir, picknumber)
    print("抽取的文件:", sample)  # 打印抽取的文件
    for name in sample:
        shutil.move(fileDir + name, tarDir + "/" + name)
    return


def movelabel(file_list, file_label_train, file_label_val):
    for i in file_list:
        if i.endswith('.jpg'):  # 修改为处理.jpg文件
            filename = file_label_train + "/" + i[:-4] + '.labels'  # 假设标签文件是.txt格式
            if os.path.exists(filename):
                shutil.move(filename, file_label_val)
                print(i + "处理成功！")


if __name__ == '__main__':
    fileDir = r"/Users/jerry/Downloads/sanchuangsai/dataset/images/train" + "/"  # 源图片文件夹路径
    tarDir = r"/Users/jerry/Downloads/sanchuangsai/dataset/images/val"  # 图片移动到新的文件夹路径
    moveimg(fileDir, tarDir)
    file_list = os.listdir(tarDir)
    file_label_train = r"/Users/jerry/Downloads/sanchuangsai/dataset/labels/train"  # 源图片标签路径
    file_label_val = r"/Users/jerry/Downloads/sanchuangsai/dataset/labels/val"  # 标签移动到新的文件路径
    movelabel(file_list, file_label_train, file_label_val)