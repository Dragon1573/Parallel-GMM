#!/bin/bash

if [[ -z $1 ]] || [[ -z $2 ]] || [[ -z $3 ]]; then
    echo "用法：testScript.sh <测试次数> <线程数量> <输出文件名>"
    exit
fi

if [ -e $3 ]; then
    rm -v $3;
fi

# 其中，$1是测试次数
for i in $(seq $1); do
    # 其中，$2是线程数量，$3是输出文件名
    { time Parallel-GMM/OpenMP/OpenMP $2 Datasets/test-200K.csv \
      Datasets/results.xml 200000 11 2; } 2>&1 | grep 'real' >> $3;
done

echo '测试完成！'
