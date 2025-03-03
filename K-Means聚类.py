from numpy import *
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import  urllib.request
import json
from time  import sleep
import time

'''
该算法会创建k个点作为聚类中心，
对数据集中的每个点计算它到各个中心的距离，
然后将点分配给距离最小的中心。再重新计算质心，
这个过程反复多次，直到数据点的簇分类结果不在改变为止。
'''

def loadDataSet():
    dataMat = []
    fr = open('testSet2.txt')
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        if not curLine or len(curLine) < 2:
            continue
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    if not dataMat:
        raise ValueError("Data set is empty after loading")
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # 计算两个向量的欧式距离

def randCent(dataSet, k):
    if dataSet.size == 0:
        raise ValueError("Data set is empty")
    
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        if dataSet[:, j].size == 0:
            raise ValueError(f"Column {j} in data set is empty")
        
        minJ = min(dataSet[:, j]).item()
        maxJ = max(dataSet[:, j]).item()
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True  # 该值为 true，则继续迭代
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0], :]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

def evaluate_kmeans(dataSet, k_values):
    sil_scores = []
    for k in k_values:
        centroids, clusterAssment = kMeans(dataSet, k)
        labels = clusterAssment[:, 0].A.flatten()
        # 将 numpy.matrix 转换为 numpy.array
        data_array = asarray(dataSet)
        labels_array = asarray(labels)
        score = silhouette_score(data_array, labels_array)
        sil_scores.append(score)
        print(f'K={k}, Silhouette Score: {score}')
    best_k = k_values[sil_scores.index(max(sil_scores))]
    print(f'Best K: {best_k} with Silhouette Score: {max(sil_scores)}')
    return best_k, sil_scores

# # 测试函数
# dataMat = mat(loadDataSet())

# # 选择 K 值范围
# k_values = range(2, 11)
# best_k, sil_scores = evaluate_kmeans(dataMat, k_values)

# # 使用最佳 K 值进行聚类
# myCentroids, myClusterAssment = kMeans(dataMat, best_k)
# print(myCentroids)
# print(myClusterAssment)

# # 可视化, 绘制散点图
# fig, ax = plt.subplots()
# colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple']
# for i in range(shape(dataMat)[0]):
#     clusterIndex = int(myClusterAssment[i, 0])
#     ax.scatter(dataMat[i, 0], dataMat[i, 1], color=colors[clusterIndex], label=f"Cluster {clusterIndex}" if i == 0 else "")
# for i in range(len(myCentroids)):
#     ax.scatter(myCentroids[i, 0], myCentroids[i, 1], color='black', marker='x', s=100, label=f"Centroid {i}")
# ax.legend(loc='upper right')

# # 绘制轮廓系数变化图
# plt.figure()
# plt.plot(k_values, sil_scores, marker='o')
# plt.xlabel('Number of Clusters (K)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score vs Number of Clusters')
# plt.show()


#轮廓系数越大就K值越好，选择k值时要选择轮廓系数大的



#二分k均值聚类算法
def biKmeans(dataSet, k, distMeans=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeans(mat(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            print(f"Cluster {i} has {ptsInCurrCluster.shape[0]} points")
            if ptsInCurrCluster.shape[0] == 0:
                print(f"Cluster {i} is empty, skipping...")
                continue
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeans)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        if bestNewCents is None:
            print("No valid clusters found, breaking the loop.")
            break
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the best cent to split is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment
# dataMat=mat(loadDataSet())
# centList,myClusterAssment=biKmeans(dataMat,3)

# # 选择 K 值范围
# k_values = range(2, 11)
# best_k, sil_scores = evaluate_kmeans(dataMat, k_values)


# # 可视化, 绘制散点图
# fig, ax = plt.subplots()
# colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple']
# for i in range(shape(dataMat)[0]):
#     clusterIndex = int(myClusterAssment[i, 0])
#     ax.scatter(dataMat[i, 0], dataMat[i, 1], color=colors[clusterIndex], label=f"Cluster {clusterIndex}" if i == 0 else "")
# for i in range(len(centList)):
#     ax.scatter(centList[i, 0], centList[i, 1], color='black', marker='x', s=100, label=f"Centroid {i}")
# ax.legend(loc='upper right')

# # 绘制轮廓系数变化图
# plt.figure()
# plt.plot(k_values, sil_scores, marker='o')
# plt.xlabel('Number of Clusters (K)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score vs Number of Clusters')
# plt.show()


# Yahoo!  palcefinder   api
# 高德地图获取地理编码
def geoGrab(stAddress, city):
    apiStem = 'https://restapi.amap.com/v3/geocode/geo?'
    address = '%s %s' % (stAddress, city)
    params = {
        'address': address,    # 输入地址
        'key': 'f16a539e8f0958a541046db5b7b9180e'  # 替换为你自己的高德API Key
    }
    url_params = urllib.parse.urlencode(params)
    amapApi = apiStem + url_params
    print(f"请求URL：{amapApi}")
    
    try:
        response = urllib.request.urlopen(amapApi)
        data = json.loads(response.read())
        print(f"API响应：{data}")
        
        if data['status'] == '1' and data['geocodes']:
            location = data['geocodes'][0]['location']  # 经纬度信息
            lat, lng = location.split(',')
            return float(lat), float(lng)
        else:
            print(f"无法获取 {address} 的经纬度，错误信息：{data.get('info', '未知错误')}")
            return None
    except Exception as e:
        print(f"请求错误：{e}")
        return None

# 处理文件中的地址
def massPlaceFind(addresses):
    with open('places.txt', 'w') as fw:
        for line in addresses:
            result = geoGrab(line[1], line[2])
            if result:
                lat, lng = result
                print('%s\t%f\t%f' % (line[0], lat, lng))
                fw.write('%s\t%f\t%f\n' % (line[0], lat, lng))
            else:
                print('error')
            time.sleep(1)  # 增加延迟，防止请求过于频繁


# 测试
# stAddress = "嘉兴市南湖区嘉兴大学（梁林校区）"
# city = "嘉兴"
# lat, lng = geoGrab(stAddress, city)
# print(f"经纬度：{lat}, {lng}")

addresses = [
    ("故宫博物院", "景山前街4号", "北京市"),
    ("天坛", "天坛路", "北京市"),
    ("颐和园", "新建宫门路19号", "北京市"),
    ("长城", "八达岭长城景区", "北京市"),
    ("鸟巢", "国家体育场北路1号", "北京市"),
    ("天安门广场", "长安街", "北京市"),
    ("圆明园", "圆明园遗址公园", "北京市"),
    ("国贸大厦", "建国门外大街1号", "北京市"),
    ("北京市大观园", "大观园路", "北京市"),
    ("北京电视台", "朝阳门南小街11号", "北京市"),
    ("北京大学", "颐和园路5号", "北京市"),
    ("清华大学", "清华园路1号", "北京市"),
    ("中国国家博物馆", "天安门广场东侧", "北京市"),
    ("798艺术区", "酒仙桥路4号", "北京市"),
    ("北京动物园", "西直门外大街137号", "北京市"),
    ("北京天文馆", "西单大街138号", "北京市"),
    ("北京昌平小汤山", "小汤山镇", "北京市"),
    ("北京欢乐谷", "东四环中路21号", "北京市"),
    ("水立方", "体育馆路11号", "北京市"),
    ("中国人民革命军事博物馆", "复兴路9号", "北京市")
]

# # 调用函数
# massPlaceFind(addresses)

# 对坐标进行聚类
def distSLC(vecA,vecB):
    R=6371.0
    a=sin(vecA[0,1]*pi/180)*sin(vecB[0,1]*pi/180)
    b=cos(vecA[0,1]*pi/180)*cos(vecB[0,1]*pi/180)*cos(pi*(vecB[0,0]-vecA[0,0])/180)
    return arccos(a+b)*R
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt', 'r', encoding='utf-8').readlines():
        lineArr = line.strip().split('\t')
        if not lineArr or len(lineArr) < 3:
            continue
        datList.append([float(lineArr[1]), float(lineArr[2])])
    if not datList:
        raise ValueError("Data set is empty after loading places.txt")
    
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeans=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', '+', 'x', 'D', 'v', '<']
    axprops = fig.add_axes(rect, label='axes0')
    imgp = plt.imread('bj5.jpg')  # 确保文件路径正确
    axprops.imshow(imgp)
    ax1 = fig.add_axes(rect, label='axes1', frameon=False)
    for i in range(numClust):
        ptsInCurClust = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurClust[:, 0].flatten().A[0], ptsInCurClust[:, 1].flatten().A[0], marker=markerStyle, s=90, color='None', edgecolor='k')
        ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300, color='k')
    plt.show()
clusterClubs()






