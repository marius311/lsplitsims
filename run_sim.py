import healpy as H
from mspec.get_kernels import quickkern
from cosmoslik import *
from numpy import *
import struct
from scipy.linalg import cho_solve, cholesky
import os, os.path as osp
from numpy.linalg import inv
from numpy.random import normal
from scipy.optimize import fmin, minimize
import cPickle as pickle
from hashlib import md5
import sys
from scipy import stats
import gc
from mspec import mpi_map, get_bin_func, mspec_log
from cosmoslik_plugins.likelihoods.clik import clik

param = param_shortcut('start','scale')

tocl=2*pi/arange(2,2509)/(arange(2,2509)+1)
fidcl=hstack([zeros(2),loadtxt("base_plikHM_TT_tau07.minimum.theory_cl")[:,1]*tocl])


class plik_lite_binned(SlikPlugin):
    """
    A binned plik_lite likelihood using the files distributed with Clik
    """    

    def __init__(self,
                 clik_folder,
                 lslice,
                 calPlanck=1,
                 sim=False):
        """
        clik_folder : path to plik_lite_v18_TT.clik
        lslice : a slice object indicating the l-range used, e.g. `slice(30,800)`
        """
        super(plik_lite_binned,self).__init__()
        root = osp.join(clik_folder,'clik','lkl_0','_external')
        self.calPlanck = calPlanck
        
        #bin specs
        self.blmin=array(loadtxt(osp.join(root,'blmin.dat')),dtype=int)
        self.blmax=array(loadtxt(osp.join(root,'blmax.dat')),dtype=int)
        self.blweight=loadtxt(osp.join(root,'bweight.dat'))

        #create bin object
        q=array([hstack([zeros(30+lmin),self.blweight[lmin:lmax+1],zeros(3000-lmax)]) for lmin,lmax in zip(self.blmin,self.blmax)[:215]])
        self.bin = get_bin_func('q',q=q)
        self.bslice=self.bin(lslice)

        #load data and covariance
        self.cl = loadtxt(osp.join(root,'cl_cmb_plik_v18.dat'))[:,1]
        f=open(osp.join(root,"c_matrix_plik_v18.dat")).read()
        cov = self.cov = array(struct.unpack('d'*(len(f)/8-1),f[4:-4])).reshape(613,613)[:215,:215]
        cov[cov==0]=cov.T[cov==0] 

        #possibly draw random spectrum
        if sim: self.cl = random.multivariate_normal(self.bin(fidcl),self.cov)

        #slice to lmin/lmax
        self.cl = self.cl[self.bslice]
        self.cov = self.cov[self.bslice,self.bslice]

        #compute decomposition
        self.cho_cov = cholesky(self.cov), False
        

    def __call__(self,cl):
        #compute likelihood
        dcl = self.bin(cl)[self.bslice]*self.calPlanck**2 - self.cl
        return dot(dcl,cho_solve(self.cho_cov,dcl))/2

        


class lowl(SlikPlugin):
    """
    A full-sky fsky scaled analytic likelihood.
    The observed Cls are generated randomly each time this is initialized. 
    """

    def __init__(self,mask_file,fidcl):

        self.mask = H.read_map(mask_file)
        self.fsky = self.mask.sum()/self.mask.size

        qk=quickkern(200)
        imll=inv(qk.get_mll(H.anafast(self.mask,lmax=60))[:60,:60])

        self.clobs = dot(imll,H.anafast(self.mask*H.synfast(fidcl,16,new=True,verbose=False),lmax=59))

    def __call__(self,cl):

        @vectorize
        def cllike(l):
            f=(2*l+1)/cl[l]/self.fsky
            return stats.chi2((2*l+1)).pdf(f*self.clobs[l])*f

        return -log(cllike(arange(2,30))).sum()

   

class planck(SlikPlugin):

    def __init__(self, lslice=slice(2,2509), sim=False, model='lcdm', 
                 action='minimize', cov='planck_2_2500.covmat',
                 highl='custom'):

        super(planck,self).__init__(**all_kw(locals()))

        self.cosmo = get_plugin('models.cosmology')(
            logA = param(3.108,0.03),
            ns = param(0.962,0.006),
            ombh2 = param(0.02221,0.0002),
            omch2 = param(0.1203,0.002),
            tau = param(0.085,0.01,min=0,gaussian_prior=(0.07,0.02)),
            H0 = param(67,1),
            pivot_scalar=0.05,
            mnu = 0.06,
        )
        self.get_cmb = get_plugin('models.camb')()
                
        self.calPlanck=param(1,0.0025,gaussian_prior=(1,0.0025))

        class clikwrap(clik):
            def __call__(self,cl):
                return super(clikwrap,self).__call__({'cl_TT':cl*arange(cl.size)*(arange(cl.size)+1)/2/pi})

        if highl=='custom':
            self.highl = plik_lite_binned(clik_folder='plik_lite_v18_TT.clik',lslice=lslice,sim=sim)
        elif highl=='plik_lite':
            assert lslice.start in [2,30] and lslice.stop==2509
            self.highl = clikwrap(clik_file='plik_lite_v18_TT.clik')
        elif highl=='plik':
            assert lslice.start in [2,30] and lslice.stop==2509
            self.highl = clikwrap(
                            clik_file='plik_v18_TT.clik',
                            A_ps_100=param(150,min=0),
                            A_ps_143=param(60,min=0),
                            A_ps_217=param(60,min=0),
                            A_cib_143=param(10,min=0),
                            A_cib_217=param(40,min=0),
                            A_sz=param(5,scale=1,range=(0,20)),
                            r_ps=param(0.7,range=(0,1)),
                            r_cib=param(0.7,range=(0,1)),
                            n_Dl_cib=param(0.8,scale=0.2,gaussian_prior=(0.8,0.2)),
                            cal_100=param(1,scale=0.001),
                            cal_217=param(1,scale=0.001),
                            xi_sz_cib=param(0.5,range=(-1,1),scale=0.2),
                            A_ksz=param(1,range=(0,5)),
                            Bm_1_1=param(0,gaussian_prior=(0,1),scale=1)
                        )
        else:
            raise ValueError(highl)

        if lslice.start==2:
            if sim:
                self.lowl=lowl(
                   mask_file="commander_dx11d2_mask_temp_n0016_likelihood_v1.fits",
                   fidcl=fidcl
                )
            else:
                self.lowl=clikwrap(clik_file='commander_rc2_v1.1_l2_29_B.clik')
        else:
            assert lslice.start>=30, "can't slice lowl likelihood"
            self.lowl=None
        

        self.priors = get_plugin('likelihoods.priors')(self)
        if action=='minimize':
            self.sampler = Minimizer(self,whiten_cov=cov)
        else:
            self.sampler = get_plugin('samplers.metropolis_hastings')(
                                self,
                                num_samples=1e6,
                                print_level=2,
                                output_file='shared/results/test_chain_%i_%i.chain'%(lslice.start,lslice.stop),
                                proposal_cov=cov,
                           )
        

    def __call__(self):
        self.cosmo.As = exp(self.cosmo.logA)*1e-10
        self.highl.A_Planck = self.highl.calPlanck = self.calPlanck
        if self.lowl: self.lowl.A_planck = self.calPlanck

        self.clTT = self.get_cmb(ns=self.cosmo.ns,
                                 As=self.cosmo.As,
                                 ombh2=self.cosmo.ombh2,
                                 omch2=self.cosmo.omch2,
                                 H0=self.cosmo.H0,
                                 tau=self.cosmo.tau,
                                 lmax=3000,
                                 **self.get('cambargs',{}))['cl_TT'][:2510]
        self.clTT[2:] *= (2*pi/arange(2,2510)/(arange(2,2510)+1))

        return lsum(lambda: self.priors(self),
                    lambda: self.highl(self.clTT),
                    lambda: self.lowl(self.clTT) if self.lowl else 0)


class Minimizer(SlikSampler):
    
    def __init__(self,params,whiten_cov=None,initial_scatter=1):
        super(Minimizer,self).__init__(**all_kw(locals(),['params']))
        self.sampled = params.find_sampled()
        self.x0 = [params[k].start for k in self.sampled]
        self.eps = [params[k].scale/10. for k in self.sampled]
    
        if whiten_cov is None: 
            self.G = identity(len(self.x0))
        else:
            with open(whiten_cov) as f:
                prop_names = f.readline().replace('#','').split()
                idxs = [prop_names.index(k) for k in self.sampled]
                self.G = cholesky(loadtxt(f)[ix_(idxs,idxs)]).T


    def sample(self,lnl):

        nfev=[0]
        def f(x,**kwargs):
            y = dot(self.G,x)
            l = lnl(*y,**kwargs)[0]
            if '--debug' in sys.argv: mspec_log(str((l,zip(self.sampled,y))))
            if '--progress' in sys.argv:
                nfev[0]+=1
                with open('shared/progress','a') as f: 
                    f.write('%.3f\n'%(0.001+nfev[0]/float(getargs('progress'))))
            return l
                    
        x0=dot(inv(self.G),self.x0)+self.initial_scatter*normal(size=len(self.x0))
        res = minimize(f,x0,method='Powell',options=dict(xtol=1e-4,ftol=1e-4,disp=False))
        yield dot(self.G,res.x), dot(self.G,x0), res




if __name__=='__main__':


    class NoDefault: pass
    def getargs(key,default=NoDefault,n=1):
        key = '--%s'%key
        if key in sys.argv:
            if n==1: return sys.argv[sys.argv.index(key)+1]
            else: return sys.argv[sys.argv.index(key)+1:sys.argv.index(key)+1+n]
        else:
            if default is NoDefault: raise KeyError(key)
            else: return default



    highl = getargs('highl','custom')


    if '--chain' in sys.argv:

        lslice = slice(*map(int,getargs('lslice',n=2)))
        p=Slik(planck(lslice=lslice,sim=False,action='chain',highl=highl))
        for _ in p.sample(): pass

    else:

        def do_run(lslice, sim=False, seed=None):
            mspec_log('Doing: %s %s'%(lslice,'sim(%s)'%seed if seed else ''))
            random.seed(seed)
            s = Slik(planck(lslice=slice(*lslice),sim=sim))
            bf,x0,res = s.sample().next()

            #postprocessing
            s.params.cambargs = {'redshifts':[0]}
            lnl,p = s.evaluate(*bf)
            pp = dict(zip(s.get_sampled(),bf))
            pp.update({'cosmo.theta':100*p.get_cmb.result.cosmomc_theta(),
                       'cosmo.ommh2':pp['cosmo.omch2']+pp['cosmo.ombh2'],
                       'cosmo.sigma8':p.get_cmb.result.get_sigma8()[0],
                       'cosmo.clamp':exp(pp['cosmo.logA'])*exp(-2*pp['cosmo.tau'])/10,
                       'lnl':lnl,
                       'highl_chi2':2*p.highl(p.clTT),
                       'highl_dof':p.highl.bslice.stop - p.highl.bslice.start,
                       'clTT':p.clTT,
                       'res':res,
                       'x0':x0,
                       'seed':seed})
            pp['cosmo.s8omm1/4']=pp['cosmo.sigma8']*(pp['cosmo.ommh2']/(pp['cosmo.H0']/100.)**2)**0.25

            mspec_log('Got: %s'%str((lslice,pp)))

            #save result to file
            result_dir = osp.join('shared/results',('sim_%s'%seed) if sim else 'real')
            os.makedirs(result_dir)
            pickle.dump([((seed,lslice),pp)],
                        open(osp.join(result_dir,'lslice_%i_%i'%lslice),'w'),protocol=2)

            return ((seed,lslice),pp)


        lslices = eval(getargs('lslices',"[]"))
        if lslices == []:
            for lsplit in range(100,2500,50)+[2509]:
                if lsplit<1700: lslices.append((lsplit,2509))
                for lmin in (2,30):
                    if lsplit>=650: lslices.append((lmin,lsplit))


        mspec_log('Doing lslices (%i): %s'%(len(lslices),lslices),rootlog=True)


        if '--real' in sys.argv:

            results = mpi_map(do_run,lslices)

        else:

            seeds = eval(getargs('seeds'))

            for seed in seeds:
                for lslice in lslices:
                    do_run(lslice,sim=True,seed=seed)
