#include "KMeans-Serial.hpp"
#include <cstdlib>
#include <cstring>
#include <fstream>
using namespace std;

int main(int argc, const char *argv[]) {
    // 语法检查
    if (argc < 6) {
        cerr << "用法：Serial-Version.exe <输入数据集> <输出标签集> "
            << "<数据量> <特征维度> <聚类簇数>" << endl;
        return EXIT_FAILURE;
    }

    const int size = atoi(argv[3]);
    const int dimensions = atoi(argv[4]);
    const int clusters = atoi(argv[5]);

    // 创建KMeans聚类模型
    KMeans *kmeans = new KMeans(dimensions, clusters);
    kmeans->setInitMode(InitMode::Randomly);

    // 载入数据集
    ifstream file;
    file.open(argv[1], ios_base::in);
    const double *const dataset = kmeans->loadFile(file, size);

    // 无监督聚类
    const int *const labels = kmeans->fit_transform(dataset, size);

    // 判断文件输出
    if (argc >= 3) {
        if (!strcmp(argv[2], "console")) {
            // 展示模型信息
            cout << kmeans << endl;
            for (int i = 0; i < size; i++) {
                cout << "(" << dataset[i * dimensions];
                for (int j = 1; j < dimensions; j++) {
                    cout << ", " << dataset[i * dimensions + j];
                }
                cout << ") = " << labels[i] << endl;
            }
        } else {
            ofstream results;
            results.open(argv[2], ios::out | ios::trunc);
            results << kmeans << endl;
            for (int i = 0; i < size; i++) {
                results << "(" << dataset[i * dimensions];
                for (int j = 0; j < dimensions; j++) {
                    results << ", " << dataset[i * dimensions + j];
                }
                results << ") = " << labels[i] << endl;
            }
            results.close();
        }
    } else {
        cerr << "警告：输出结果被忽略！" << endl;
    }

    // 释放数组占用的内存
    delete[] dataset;
    delete[] labels;

    return EXIT_SUCCESS;
}
