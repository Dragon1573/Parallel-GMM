#include "KMeans-Serial.hpp"
#include <cstdlib>
#include <cstring>
#include <fstream>
using namespace std;

int main(int argc, const char *argv[]) {
    // �﷨���
    if (argc < 6) {
        cerr << "�÷���Serial-Version.exe <�������ݼ�> <�����ǩ��> "
            << "<������> <����ά��> <�������>" << endl;
        return EXIT_FAILURE;
    }

    const int size = atoi(argv[3]);
    const int dimensions = atoi(argv[4]);
    const int clusters = atoi(argv[5]);

    // ����KMeans����ģ��
    KMeans *kmeans = new KMeans(dimensions, clusters);
    kmeans->setInitMode(InitMode::Randomly);

    // �������ݼ�
    ifstream file;
    file.open(argv[1], ios_base::in);
    const double *const dataset = kmeans->loadFile(file, size);

    // �޼ල����
    const int *const labels = kmeans->fit_transform(dataset, size);

    // �ж��ļ����
    if (argc >= 3) {
        if (!strcmp(argv[2], "console")) {
            // չʾģ����Ϣ
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
        cerr << "���棺�����������ԣ�" << endl;
    }

    // �ͷ�����ռ�õ��ڴ�
    delete[] dataset;
    delete[] labels;

    return EXIT_SUCCESS;
}
