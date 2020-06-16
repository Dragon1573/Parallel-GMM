import numpy
import sys

if __name__ == "__main__":
    """ 主函数 """

    if len(sys.argv) < 4:
        print("用法：generator.py <每类数据量> <维度> <类簇数量>")
        exit(1)

    clusters = []
    for i in range(int(sys.argv[3])):
        mu = float(input("类簇均值："))
        sigma = float(input("类簇方差："))
        clusters.append(
            mu + sigma * numpy.random.randn(int(sys.argv[1]), int(sys.argv[2]))
        )
        pass

    dataset = numpy.concatenate(clusters, 0)

    filename = input("数据集文件名：")
    numpy.savetxt(filename, dataset, fmt="%f",
                  delimiter="\t", encoding="UTF-8")
