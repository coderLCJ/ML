import knn

group, labels = knn.createDataSet()
print(knn.classfiy0([100, 3], group, labels, 3))



'''
d1 = 78
d2 = 78

# 计算距离
k = 3
love = 0
kongfu = 0
dist = []
for t1, t2 in group:
    # print(t1, t2)
    dist.append(pow((t1 - d1)**2 + (t2 - d2)**2, 0.5))
sortDist = sorted(dist)
print(dist)
print(sortDist)
for value in dist:
    index = sortDist.index(value)
    i = dist.index(value)
    if index < k:
        if labels[i] == '爱情片':
            love += 1
        else:
            kongfu += 1
        print(labels[i], dist[i])

if love > kongfu:
    print('这是一部爱情片')
else:
    print('这是一部功夫片')
'''