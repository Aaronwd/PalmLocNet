import xml.dom.minidom as xmldom
import os

trainpath = '/home/aaron/桌面/re-train/'
os.chdir(trainpath)
trainfilenames = os.listdir()
for i in range(len(trainfilenames)):
    trainfilenames[i] = trainpath + trainfilenames[i]
print(trainfilenames)

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
    f = open('./train.txt', 'w')
    for k in trainfilenames:
        path, xmin, ymin, xmax, ymax = parse_xml(k)
        print(path, xmin, ymin, xmax, ymax)
        f.write(path+' '+xmin+' '+ymin+' '+xmax+' '+ymax+'\n')
    f.close()



