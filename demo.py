import numpy
import scipy.io as sio
import urllib
import matplotlib.pyplot as plt
import urllib.request

data = sio.loadmat('outputbest.mat')['bestsummary'].ravel()
urls = [ i for j in range(len(data)) for i in data[j] ]

i = 1
for url in urls:
    f = open(r"pic" + '/' + str(i) + '.jpg', 'wb')
    req = urllib.request.urlopen(url)
    buf = req.read()
    t = 250+i
    f.write(buf)
    i+=1
plt.show()