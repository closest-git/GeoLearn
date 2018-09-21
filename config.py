#coding:utf8
import warnings
#from pytorch_env import *
import numpy.polynomial.chebyshev as cheb

class DefaultConfig(object):
    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        # if len(kwargs)>0:
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        #for k, v in self.__class__.__dict__.items():
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print("\t{}:    {}".format(k, getattr(self, k)))


    def Init_lenda_tic(self ):
        l_0 = self.lenda_0

        self.lenda_tic=[]
        if self.tic_type=="cheb_":
            pt_1 = cheb.chebpts1(256)
            pt_2 = cheb.chebpts2(256)
            scale = (self.lenda_1-l_0)/2
            off = l_0+scale
            self.lenda_tic = [i * scale + off for i in pt_2]
            assert(self.lenda_tic[0]==l_0 and self.lenda_tic[-1]==self.lenda_1)
        else:

            if( self.lenda_0<500 ):
                self.lenda_tic = list(range(self.lenda_0,500,1))
                l_0 = 500
            self.lenda_tic.extend( list(range(l_0,self.lenda_1+self.lenda_step,self.lenda_step)) )

    def __init__(self,fix_seed=None):
        self.env = 'default' # visdom 环境
        #self.device = pytorch_env(fix_seed)
        self.model = 'ResNet34' # 使用的模型，名字必须与models/__init__.py中的名字一致
        self.use_gpu = True

        self.nLayer = 10
        self.thick_0 = 5
        self.thick_1 = 50

        self.tic_type = "cheb_"
        # self.tic_type = "cheb_2"
        self.lenda_0 = 240      #240
        # self.lenda_1 = 260
        self.lenda_1 = 2000
        self.lenda_step = 5
        self.Init_lenda_tic()

        self.noFeat4CMM = 1     #  CMM用来计算3个系数 0,1,2   三条曲线
        self.normal_Y = 0       #反而不行，需要对MLP重新认识

        self.dump_NN = True
        self.n_dict = None
        self.user_loss = None




