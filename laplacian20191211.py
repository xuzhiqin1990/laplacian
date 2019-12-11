#author: Zhiqin Xu 许志钦
#email: xuzhiqin@sjtu.edu
#2019-09-24
# coding: utf-8


import os
import matplotlib
matplotlib.use('Agg')   
import pickle
import time  
import shutil 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt   
#from BasicFunc import mySaveFig, mkdir

from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


Leftp=0.18
Bottomp=0.18
Widthp=0.88-Leftp
Heightp=0.9-Bottomp
pos=[Leftp,Bottomp,Widthp,Heightp]

def mkdir(fn):
    if not os.path.isdir(fn):
        os.mkdir(fn)
    

def mySaveFig(pltm, fntmp,fp=0,ax=0,isax=0,iseps=0,isShowPic=0):
    if isax==1:
        #pltm.legend(fontsize=18)
        # plt.title(y_name,fontsize=14)
        #ax.set_xlabel('step',fontsize=18)
        #ax.set_ylabel('loss',fontsize=18)
        pltm.rc('xtick',labelsize=18)
        pltm.rc('ytick',labelsize=18)
        ax.set_position(pos, which='both')
    fnm='%s.png'%(fntmp)
    pltm.savefig(fnm)
    if iseps:
        fnm='%s.eps'%(fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp!=0:
        fp.savefig("%s.pdf"%(fntmp), bbox_inches='tight')
    if isShowPic==1:
        pltm.show() 
    elif isShowPic==-1:
        return
    else:
        pltm.close()


        

R={}  ### used for saved all parameters and data
 
### mkdir a folder to save all output
sBaseDir = 'fitnd6/'
#import platform
#if platform.system()=='Windows':
#    BaseDir = 'D:/nn/%s'%(sBaseDir)
#else:
#    BaseDir = sBaseDir
    
BaseDir = sBaseDir
    
subFolderName = '%s'%(int(np.absolute(np.random.normal([1])*100000))//int(1)) 
FolderName = '%s%s/'%(BaseDir,subFolderName)
mkdir(BaseDir)
mkdir(FolderName)
mkdir('%smodel/'%(FolderName))
R['FolderName']=FolderName 
R['input_dim']=2

R['output_dim']=1
R['hidden_units']=[200,20,20]
R['learning_rate']=2e-5
R['learning_rateDecay']=5e-7
R['issavemodel']=False

plotepoch=500
R['x_in_train_size']=1000  ### training size
#R['x_in_test_size']=500  ### test size

R['x_b_train_size']=25;  ### training size 
#R['x_b_test_size']=25  ### test size

R['beta']=1000


R['tol']=1e-4
R['Total_Step']=600000  ### the training step. Set a big number, if it converges, can manually stop training 

R['FolderName']=FolderName   ### folder for save images



    

    
def get_rand_x(n_size,dim,xrange=[0,1]):
    xxs=np.random.rand(n_size,dim)*(xrange[1]-xrange[0])+xrange[0]
    return xxs

def get_x_boundary(n_size,dim,xrange=[0,1]):
    if dim==1:
        return np.reshape(xrange,[-1,1])
    xx_b=get_rand_x(n_size*dim*2,dim,xrange)
    for ii in range(dim):
        xx_b[2*ii*n_size:(2*ii+1)*n_size,ii]=xrange[0]        
        xx_b[(2*ii+1)*n_size:2*(ii+1)*n_size,ii]=xrange[1]
    
    return xx_b

def get_target_func(xx_in):
    y_tmp=0
    y_tmp+=np.sum(xx_in**2,axis=1,keepdims=True)
    
    return y_tmp

def get_grad2_target(xx_in):
    return np.ones([np.shape(xx_in)[0],1])*2*R['input_dim']
    
def get_xy(n_size_in, n_size_b):
    x_inside=get_rand_x(n_size_in,R['input_dim'])
    y_inside=get_target_func(x_inside)
    
    xx_b=get_x_boundary(n_size_b,R['input_dim'])
    yy_b=get_target_func(xx_b)
    
    g2d_yy_in=get_grad2_target(x_inside)
    return x_inside,y_inside,xx_b,yy_b,g2d_yy_in



def add_layer2(x,input_dim = 1,output_dim = 1, name_scope='hidden'):
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        ua_w = tf.get_variable(name='ua_w', shape=(input_dim,output_dim), dtype=tf.float32)
        ua_b = tf.get_variable(name='ua_b', shape=(output_dim,), dtype=tf.float32) 
        z=tf.matmul(x, ua_w)+ua_b
        output_z = tf.nn.tanh(z)
        
    return output_z


def univAprox2(x0, hidden_units=[10,20,40],input_dim = 1,output_dim_final = 1):
   
    hidden_num = len(hidden_units)
    add_hidden = [input_dim] + hidden_units;
    output=x0
    for i in range(hidden_num):
        input_dim = add_hidden[i]
        output_dim = add_hidden[i+1]
        print('input_dim:%s, output_dim:%s'%(input_dim,output_dim))
        name_scope = 'hidden' + np.str(i+1)
        output=add_layer2(output,input_dim,output_dim,name_scope= name_scope)
    
    ua_we = tf.get_variable(name='ua_we', shape=(add_hidden[-1],output_dim_final), dtype=tf.float32)
    ua_be = tf.get_variable(name='ua_be', shape=(output_dim_final,), dtype=tf.float32)
    
    z = tf.matmul(output, ua_we)+ua_be
    return z



def get_sample():
    R['train_x_inside'],R['train_y_inside'],R['train_xx_b'],R['train_yy_b'],R['train_g2d_yy_in']=get_xy(R['x_in_train_size'], R['x_b_train_size'])
    #R['test_x_inside'],R['test_y_inside'],R['test_xx_b'],R['test_yy_b'],R['test_g2d_yy_in']=get_xy(R['x_in_test_size'], R['x_b_test_size'])


#tf.compat.v1.reset_default_graph()
with tf.variable_scope('Graph',reuse=tf.AUTO_REUSE) as scope:
    # Our inputs will be a batch of values taken by our functions
    # x is the interior points
    x = tf.placeholder(tf.float32, shape=[None, R['input_dim']],name="x")
    # x_len is the length of x 
    x_len= tf.placeholder_with_default(input=1,shape=[],name='xlen')
    # xb is the sample on boundary
    xb = tf.placeholder_with_default(input=
                    np.float32(np.zeros([1,R['input_dim']])), 
                    shape=[None, R['input_dim']], name="xb")
    # y_true is the label of interior samples
    y_true = tf.placeholder_with_default(input=[[0.0]], 
                    shape=[None, R['output_dim']], name="y")
    # y_true_b is the label of the sample on boundary
    y_true_b = tf.placeholder_with_default(input=[[0.0]], 
                    shape=[None, R['output_dim']],name="yb")
    # tg2u is the true 2nd gradient of interior samples
    tg2u = tf.placeholder_with_default(input=[[0.0]], 
                    shape=[None, R['output_dim']],name="tg2u")
    # beta is the penality weight of boundary
    beta= tf.placeholder_with_default(input=1.0,shape=[],name='beta')
    # _lr is the learning rate of each step.
    _lr= tf.placeholder_with_default(input=1e-3,shape=[],name='lr')
    #Laplacian
    # To compute laplacian, we need to unstack of x, such that
    # each dimension is one variable
    _x=tf.unstack(x,axis=1) 
    #unstack each dimension, so each dim is a variable
    # We pack all dim together so we can get their predict values
        # first, we need to expand each dim to shape of [-1,1]
        # then, pack them as the final shape as [-1,1]
    _x_=[tf.expand_dims(tmp,axis=1) for tmp in _x]  
    _x2=tf.transpose(tf.stack(_x_))[0]
    # all inputs combines both the interior and boundary samples.
    _x_0=tf.concat([_x2,xb],axis=0)
    _y=tf.concat([y_true,y_true_b],axis=0)
    # univAprox2 is the DNN that returns predicted labels
    y= univAprox2(_x_0, R['hidden_units'],
                  input_dim = R['input_dim'],
                  output_dim_final=R['output_dim'])
    
    # only consider the prediected values of 
    # interior points, that is, y[0:x_len].
    # the shape of _grady is [n_dim, x_len]
    _grady=[tf.squeeze(tf.gradients(y[0:x_len],tmp)) for tmp in _x]
    # the shape of grady is [x_len,n_dim]
    #grady= tf.transpose(tf.stack(_grady) )
    grad2y=0
    for ii in range(R['input_dim']):
        # take out each dimension and do grad
        grad2y+=tf.stack(tf.gradients(_grady[ii],_x_[ii]))[0]
    loss0=tf.reduce_mean(tf.square(grad2y-tg2u))
    lossb=tf.reduce_mean(tf.square(y_true_b-y[x_len:]))
    loss=loss0+beta*lossb
    lossy=tf.reduce_mean(tf.square(_y-y))
    # We define our train operation using the Adam optimizer
    adam = tf.compat.v1.train.AdamOptimizer(learning_rate=_lr)
    train_op = adam.minimize(loss)
    
config = tf.ConfigProto()
config.gpu_options.allow_growth=True  
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()  

#get_sample()
#_gradyt,grad2yt= sess.run([_grady,grad2y],feed_dict={x: R['test_x_inside'],x_len:np.shape(R['test_x_inside'])[0],
#                                                xb:R['test_xx_b'], y_true: R['test_y_inside'], y_true_b:R['test_yy_b'],
#                                                tg2u: R['test_g2d_yy_in'], beta: R['beta']})
#print(np.shape(_gradyt[ii]))
#print(np.shape(grad2yt))
#print(np.shape(_gradyt))
    
    
t0=time.time()       
class model():
    def __init__(self): 
        R['y_train']=[] 
        R['loss_train']=[] 
        R['lossy_train']=[]
        if R['issavemodel']:
            nametmp='%smodel/'%(FolderName)
            mkdir(nametmp)
            saver.save(sess, "%smodel.ckpt"%(nametmp))
        
    def run_onestep(self):
        get_sample() 
        R['y_train'], lossy_train_tmp, loss_train_tmp= sess.run([y, lossy,loss],feed_dict={x: R['train_x_inside'],x_len:np.shape(R['train_x_inside'])[0],
                                                xb:R['train_xx_b'], y_true: R['train_y_inside'], y_true_b:R['train_yy_b'],
                                                tg2u: R['train_g2d_yy_in'], beta: R['beta']}) 
        R['loss_train'].append(loss_train_tmp) 
        R['lossy_train'].append(lossy_train_tmp)
        _= sess.run(train_op,feed_dict={x: R['train_x_inside'],x_len:np.shape(R['train_x_inside'])[0],
                                                xb:R['train_xx_b'], y_true: R['train_y_inside'], y_true_b:R['train_yy_b'],
                                                tg2u: R['train_g2d_yy_in'], beta: R['beta'],_lr:R['learning_rate']})
        R['learning_rate']=R['learning_rate']*(1-R['learning_rateDecay'])
        
    def run(self,step_n=1):
        if R['issavemodel']:
            nametmp='%smodel/model.ckpt'%(FolderName)
            saver.restore(sess, nametmp)
        for ii in range(step_n):
            self.run_onestep()
            if R['loss_train'][-1]<R['tol']:
                print('model end, error:%s'%(R['lossu_train'][-1]))
                self.plotloss()
                self.savefile()
                
                if R['issavemodel']:
                    nametmp='%smodel/'%(FolderName)
                    shutil.rmtree(nametmp)
                    saver.save(sess, "%smodel.ckpt"%(nametmp))
                break
                
            if ii%plotepoch==0:
                print('time elapse: %.3f'%(time.time()-t0)) 
                print('model, epoch: %d, train loss: %f' % (ii,R['loss_train'][-1])) 
                print('model, epoch: %d, train lossy: %f' % (ii,R['lossy_train'][-1]))
                self.plotloss()
                self.savefile()
                if R['issavemodel']:
                    nametmp='%smodel/'%(FolderName)
                    shutil.rmtree(nametmp)
                    saver.save(sess, "%smodel.ckpt"%(nametmp))
            
                
    def plotloss(self):
        plt.figure()
        ax = plt.gca() 
        y2 = R['loss_train'] 
        plt.plot(y2,'g*',label='Train')
        ax.set_xscale('log')
        ax.set_yscale('log')                
        plt.legend(fontsize=18)
        plt.title('loss',fontsize=15)
        fntmp = '%sloss'%(FolderName)
        mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
        
        plt.figure()
        ax = plt.gca() 
        y2 = R['lossy_train'] 
        plt.plot(y2,'g*',label='Train')
        ax.set_xscale('log')
        ax.set_yscale('log')                
        plt.legend(fontsize=18)
        plt.title('lossy',fontsize=15)
        fntmp = '%slossy'%(FolderName)
        mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
        
        if R['input_dim']==1:
            plt.figure()
            ax = plt.gca()
            plt.plot(R['train_x_inside'],R['y_train'][0:R['x_in_train_size'],:],'ro',label='train')
            plt.plot(R['train_x_inside'],R['train_y_inside'][0:R['x_in_train_size'],:],'g*',label='True')
            #ax.set_xscale('log')
            #ax.set_yscale('log')                
            plt.legend(fontsize=18)
            plt.title('y',fontsize=15)
            fntmp = '%sy'%(FolderName)
            mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
            
        if R['input_dim']==2:
            
            fp = plt.figure()
            ax = fp.gca(projection='3d')
            ax.scatter(R['train_x_inside'][:,0], R['train_x_inside'][:,1],R['train_y_inside'],label='true')
            ax.scatter(R['train_x_inside'][:,0], R['train_x_inside'][:,1],R['y_train'][0:R['x_in_train_size'],:],label='dnn')
            # Customize the z axis.
            #ax.set_zlim(-2.01, 2.01)
            ax.zaxis.set_major_locator(LinearLocator(5))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            plt.legend(fontsize=18)
            fntmp = '%sy'%(FolderName)
            mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
            
    def savefile(self):
        with open('%s/objs.pkl'%(FolderName), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(R, f, protocol=4)
         
        text_file = open("%s/Output.txt"%(FolderName), "w")
        for para in R:
            if np.size(R[para])>20:
                continue
            text_file.write('%s: %s\n'%(para,R[para]))
        
        text_file.close()
            
model1=model()
model1.run(100000)
    
    
