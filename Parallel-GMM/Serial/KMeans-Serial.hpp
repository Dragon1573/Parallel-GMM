#pragma once
#include <iostream>
using namespace std;

// ��������ʼ��ģʽ
enum InitMode {
    Randomly, // �����ʼ��
    Manually, // ָ��ֵ��ʼ��
    Uniformly // ���ȳ�ʼ��
};

/**
 @breif K��ֵ�㷨

 @date 2003/10/16 Fei Wang
 @date 2013       Jave Lu     ������¬��
 @date 2020/05/25 Jeff Huang  ���ٶ�������ߣ�
 */
class KMeans {
public:
    /**
     * @brief ���캯��
     *
     * @param dimensions    ����ά��
     * @param clusters      ���������
     * @return
    */
    KMeans(const int dimensions = 1, const int clusters = 1);
    // ��������
    ~KMeans();

    /**
     * @brief ���þ�����������
     *
     * @param i     �����
     * @param point �����
    */
    void setMean(const int i, const double *point);
    // ���ó�ʼ��ģʽ
    void setInitMode(enum InitMode mode);
    // ��������������
    void setMaxIterations(const int iterations);
    // ����������
    void setEpsilon(const double epsilon);
    // ��ȡ��������
    const double *const getMean(const int i);
    // ��ȡ��ʼ��״̬
    const int getInitMode();
    // ��ȡ����������
    const int getMaxIterations();
    // ��ȡ������
    const double getEpsilon();

    /**
     * @brief ��ģ�ͽ��о���
     *
     * @param datasets      ���ݼ���һά���飬��������ģ���ά����
     * @param dataSize    ������
    */
    const int *fit_transform(const double *const datasets, const int dataSize);
    /**
     * @brief ���ļ��������ݼ�
     *
     * @param  file          ���ݼ��ļ���
     * @param  dimensions    ����ά��
     * @param  size          ������
     * @return ���ݼ�����
    */
    const double *const loadFile(ifstream &file, const int size);

    /**
     * @brief �޼ල����Ԥ��
     *
     * @param dataset
     * @param dataSize
    */
    const int *predict(const double *const dataset, const int dataSize);

    /**
     * @brief ���ģ����Ϣ
     *
     * @param out       �����
     * @param kmeans    K�����ľ����ģ��
    */
    friend ostream &operator<<(ostream &, const KMeans *const);

private:
    // ά������
    int dimensions;
    // ���������
    int clusters;
    // �������ľ���
    double **means;
    // ��������ʼ��ģʽ
    InitMode initMode;
    // ������������������������
    int maxInterations;
    // �����ޣ�С�ڼ�������
    double epsilon;

    /**
     * @brief ��ȡ��ǩ
     *
     * @param sample    ���������
    */
    const double getCost(const double *sample, int &label);
    /**
     @brief �������ݵ�֮���ŷ����þ���

     @param x           �����A
     @param y           �����B
     @param dimensions  �ռ�ά��
     */
    const double getDistance(const double *, const double *, const int);
};
