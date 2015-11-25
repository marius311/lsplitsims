from cosmoslik import *
from numpy import *
import struct
from scipy.linalg import cho_solve
import os.path as osp
from numpy.linalg import cholesky, inv
from scipy.optimize import fmin
import cPickle as pickle
from hashlib import md5
import sys
from scipy import stats
import gc
from mspec import mpi_map, get_bin_func, mspec_log
from cosmoslik_plugins.likelihoods.clik import clik

param = param_shortcut('start','scale')


class plik_lite_binned(SlikPlugin):
    """
    A binned plik_lite likelihood using the files distributed with Clik
    """    

    def __init__(self,clik_folder,lslice,cl=None):
        """
        clik_folder : path to plik_lite_v18_TT.clik
        lslice : a slice object indicating the l-range used, e.g. `slice(30,800)`
        """
        super(plik_lite_binned,self).__init__()
        root = osp.join(clik_folder,'clik','lkl_0','_external')

        #bin specs
        self.blmin=array(loadtxt(osp.join(root,'blmin.dat')),dtype=int)
        self.blmax=array(loadtxt(osp.join(root,'blmax.dat')),dtype=int)
        self.blweight=loadtxt(osp.join(root,'bweight.dat'))

        #create bin object
        q=array([hstack([zeros(30+lmin),self.blweight[lmin:lmax+1],zeros(3000-lmax)]) for lmin,lmax in zip(self.blmin,self.blmax)[:215]])
        self.bin = get_bin_func('q',q=q)
        self.bslice=self.bin(lslice)

        #load data and covariance
        self.cl = cl or loadtxt(osp.join(root,'cl_cmb_plik_v18.dat'))[self.bslice,1]
        f=open(osp.join(root,"c_matrix_plik_v18.dat")).read()
        cov = self.cov = array(struct.unpack('d'*(len(f)/8-1),f[4:-4])).reshape(613,613)[self.bslice,self.bslice]
        cov[cov==0]=cov.T[cov==0] #actually cholesky works fine without this, but do it for aesthetics... 
        self.cho_cov = cholesky(self.cov), False
        

    def __call__(self,cl):
        #compute likelihood
        dcl = self.bin(cl)[self.bslice] - self.cl
        return dot(dcl,cho_solve(self.cho_cov,dcl))/2

        


class lowl(SlikPlugin):
    """
    A full-sky fsky scaled analytic likelihood.
    The observed Cls are generated randomly each time this is initialized. 
    """

    def __init__(self,mask_file,fidcl):

        import healpy as H
        from mspec.get_kernels import quickkern

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

    def __init__(self, lslice=slice(2,2509), sim=True, fidcl=None, cl=None, model='lcdm'):
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
        self.priors = get_plugin('likelihoods.priors')(self)
                

        self.highl = plik_lite_binned(
            clik_folder='plik_lite_v18_TT.clik',
            lslice=lslice,
            cl=cl
        )

        if lslice.start==2:
            if sim:
                self.lowl=lowl(
                   mask_file="commander_dx11d2_mask_temp_n0016_likelihood_v1.fits",
                   fidcl=fidcl
                )
            else:
                class clikwrap(clik):
                    def __call__(self,cl):
                        return super(clikwrap,self).__call__({'cl_TT':cl*arange(cl.size)*(arange(cl.size)+1)/2/pi})
                self.lowl=clikwrap(
                    clik_file='commander_rc2_v1.1_l2_29_B.clik',
                    A_planck=1
                )
        else:
            assert lslice.start>=30, "can't slice lowl likelihood"
            self.lowl=None
        

        self.sampler = Minimizer(self)
        

    def __call__(self):
        self.cosmo.As = exp(self.cosmo.logA)*1e-10

        self.clTT = self.get_cmb(ns=self.cosmo.ns,
                                 As=self.cosmo.As,
                                 ombh2=self.cosmo.ombh2,
                                 omch2=self.cosmo.omch2,
                                 H0=self.cosmo.H0,
                                 tau=self.cosmo.tau,
                                 lmax=3000)['cl_TT'][:2509]
        self.clTT[2:] *= (2*pi/arange(2,2509)/(arange(2,2509)+1))

        return lsum(lambda: self.priors(self),
                    lambda: self.highl(self.clTT),
                    lambda: self.lowl(self.clTT) if self.lowl else 0)


class Minimizer(SlikSampler):
    
    def __init__(self,params):
        super(Minimizer,self).__init__(**all_kw(locals(),['params']))
        self.sampled = params.find_sampled()
        self.x0 = [params[k].start for k in self.sampled]
    
    def sample(self,lnl):
                
        def f(x,**kwargs):
            l = lnl(*x,**kwargs)[0]
            mspec_log(str((l,x)))
            return l
                    
        res = fmin(f,self.x0,xtol=0.0001,disp=False)
        yield res


tocl=2*pi/arange(2,2509)/(arange(2,2509)+1)
fidcl=hstack([zeros(2),loadtxt("base_plikHM_TT_tau07.minimum.theory_cl")[:,1]*tocl])


if __name__=='__main__':

    work = []
    for lmin in (2,30):
        for lsplit in range(600,2550,50):
            work.append((lmin,lsplit))
            if lsplit<1700: work.append((lsplit,2509))
    mspec_log('All work: %s'%work,rootlog=True)

    if '--real' in sys.argv:

        cl = 'cl_cmb_plik_bin1_v18.dat'
        results = {}
                
        def do_run(lslice):
            mspec_log('Doing: %s'%str(lslice))
            p=Slik(planck(fidcl=fidcl,lslice=slice(*lslice),sim=False))
            bf = p.sample().next()
            return (lslice,bf)

        results = mpi_map(do_run,work)
        pickle.dump(results,open('results/result_real','w'),protocol=2)


    else:

        highl = plik_lite(
            cov='../../remote_data/cmbcls/c_matrix_plik_bin1_v18.dat',
            cl='../../remote_data/cmbcls/cl_cmb_plik_bin1_v18.dat',
            lslice=slice(30,2509)
        )

        while True:

            cl = hstack([zeros(30),random.multivariate_normal(fidcl[highl.lslice],highl.cov)])

            def do_run(lslice):
                mspec_log('Doing: %s'%str(lslice))
                p=Slik(planck(fidcl=fidcl,lslice=slice(*lslice),cl=cl[slice(*lslice)],sim=True))
                bf = p.sample().next()
                return (lslice,bf)


            print 'Starting...'

            p1=Slik(planck(fidcl=fidcl,lslice=slice(30,800),cl=cl[30:800],lowl='sim'))
            lowbf = p1.sample().next()
            print 'low: '+str(lowbf)

            p2=Slik(planck(fidcl=fidcl,lslice=slice(30,800),cl=cl[30:800]))
            midbf = p2.sample().next()
            print 'mid: '+str(midbf)

            p3=Slik(planck(fidcl=fidcl,lslice=slice(800,2509),cl=cl[800:2509]))
            highbf = p3.sample().next()
            print 'high: '+str(highbf)


            pickle.dump(
                {
                    'low':lowbf,
                    'mid':midbf,
                    'high':highbf,
                    'high_cl':cl,
                    'low_cl':p1.params.lowl.clobs
                },
                open('results/result_'+md5(str(cl)).hexdigest(),'w'),
                protocol=2
            )

            del p1,p2,p3
            gc.collect()
