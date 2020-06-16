#!/usr/bin/env python

import sys
import re

if len(sys.argv) != 2:
    print('Usage: averageTime.py <Log_File>')
    sys.exit(1)

pattern = re.compile(r'\d+')
with open(sys.argv[1], 'r') as file:
    data = file.readlines()
    pass
data = [pattern.findall(line) for line in data]
data = [[float(x) for x in line] for line in data]
data = [60 * x[0] + x[1] + x[2] / 1000 for x in data]
print('Average Time: ' + str(sum(data) / len(data)) + 'sec')
