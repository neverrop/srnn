import numpy as np
import matplotlib.pyplot as plt

d=2; n=1000
x=np.matrix(np.random.randn(n,d))
s=np.matrix([[1., 0.],[0., 2.]])
r=np.matrix([[np.cos(np.pi/3), -np.sin(np.pi/3)],[np.sin(np.pi/3), np.cos(np.pi/3)]])

t=np.matrix([0.5, -1.])
x=x.dot(s).dot(r)+np.ones([n,1]).dot(t)
m=np.mean(x,axis=0);
cov=(x-np.ones([n,1]).dot(m)).T.dot(x-np.ones([n,1]).dot(m))/n
icov=np.linalg.inv(cov)
xx,yy=np.meshgrid(np.linspace(-5,5),np.linspace(-5,5))
xt=xx-m[0,0]
yt=yy-m[0,1]
p=1./(2.*np.pi*np.sqrt(np.linalg.det(cov))) * np.exp(-1./2.*(icov[0,0]*xt*xt+(icov[0,1]+icov[1,0])*xt*yt+icov[1,1]*yt*yt))
plt.figure()
plt.scatter(x[:,0],x[:,1]);
plt.contour(xx,yy,p,cmap='hsv');



ed,ev=np.linalg.eig(cov)
x1=x.dot(ev)
m1=np.mean(x1,axis=0)
xt1=xx-m1[0,0]
yt1=yy-m1[0,1]
a=ev.T.dot(cov).dot(ev)
ia=np.linalg.inv(a)
p1=1./(2.*np.pi*np.sqrt(np.linalg.det(a))) * np.exp(-1./2.*(ia[0,0]*xt1*xt1+(ia[0,1]+ia[1,0])*xt1*yt1+ia[1,1]*yt1*yt1))
plt.figure()
plt.scatter(x1[:,0],x1[:,1])
plt.contour(xx,yy,p1,cmap='hsv')
plt.show()
