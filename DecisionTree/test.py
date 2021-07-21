import operator

L = {'1': 1, '2': 0}
print(L)
sortedClassCount = sorted(L.items(), key=operator.itemgetter(1), reverse=True)
print(sortedClassCount)