import xml.dom.minidom as xmldom
import os

#设定路径
trainpath = '/home/aaron/桌面/PalmLocNet/picture/trainxml/'
testpath = '/home/aaron/桌面/PalmLocNet/picture/testxml/'

#收集trainset的文件名和路径
os.chdir(trainpath)
trainfilenames = os.listdir()
tfilenames = trainfilenames
print(tfilenames)
for i in range(len(trainfilenames)):
    trainfilenames[i] = trainpath + trainfilenames[i]
print(trainfilenames)

#收集testset的文件名和路径
os.chdir(testpath)
testfilenames = os.listdir()
tefilenames = testfilenames
print(tefilenames)
for i in range(len(testfilenames)):
    testfilenames[i] = testpath + testfilenames[i]
print(testfilenames)

def parse_xml(fn):
    xml_file = xmldom.parse(fn)
    eles = xml_file.documentElement
    xmin = eles.getElementsByTagName("xmin")[0].firstChild.data
    xmax = eles.getElementsByTagName("xmax")[0].firstChild.data
    ymin = eles.getElementsByTagName("ymin")[0].firstChild.data
    ymax = eles.getElementsByTagName("ymax")[0].firstChild.data
    path = eles.getElementsByTagName("path")[0].firstChild.data
    return path, xmin, ymin, xmax, ymax

if __name__ == "__main__":
    #将图片的路径和标定的坐标保存到txt文件中

    #保存train的信息
    f = open('./train.txt', 'w')
#   for (l,k) in zip(tfilenames, trainfilenames):
    for k in trainfilenames:
        path, xmin, ymin, xmax, ymax = parse_xml(k)
        print(path, xmin, ymin, xmax, ymax)
        f.write(path+' '+xmin+' '+ymin+' '+xmax+' '+ymax+'\n')
    f.close()

    #保存test的信息
    f = open('./test.txt', 'w')
    for k in testfilenames:
        path, xmin, ymin, xmax, ymax = parse_xml(k)
        print(path, xmin, ymin, xmax, ymax)
        f.write(path+' '+xmin+' '+ymin+' '+xmax+' '+ymax+'\n')
    f.close()