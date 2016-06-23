import healpy as H
from mspec.get_kernels import quickkern
from cosmoslik import *
from numpy import *
import struct
from scipy.linalg import cho_solve, cholesky
import os, os.path as osp
from numpy.linalg import inv, norm
from numpy.random import normal
from scipy.optimize import fmin, minimize
import cPickle as pickle
from hashlib import md5
import sys
from scipy import stats
import gc
from mspec import mpi_map, get_bin_func, mspec_log
from cosmoslik_plugins.likelihoods.clik import clik
import argparse
from random import randint
import pyfits
from glob import glob
import re

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




class lowl_approx(SlikPlugin):
    """
    A full-sky fsky scaled analytic likelihood.
    The observed Cls can be generated randomly based on a fiducial Cl,
    or a map can be provided.
    """

    def __init__(self,mask_file=None,f=None,maps=None,bl=None,fidcl=None):

        mask = ones(12*16**2) if mask_file is None else H.read_map(mask_file)

        self.fsky = mask.sum()/mask.size
        self.f = ones(30) if f is None else f

        qk=quickkern(200)
        imll=inv(qk.get_mll(H.anafast(mask,lmax=60))[:49,:49])

        if maps is None:
            mp1=mp2=H.synfast(fidcl,16,new=True,verbose=False)
            bl=ones(50)
        else:
            mp1,mp2=(H.ud_grade(H.read_map(m),16)*1e6 for m in maps)

        self.clobs = dot(imll,H.anafast(mask*mp1,mask*mp2,lmax=48))/bl[:49]**2
        self.chi2s = [stats.chi2((2*l+1)*self.fsky*self.f[l]) for l in range(30)]

    def __call__(self,cl):
        return -log([self.chi2s[l].pdf((2*l+1)*self.fsky*self.f[l]*self.clobs[l]*self.A_planck**2/cl[l])/cl[l] for l in range(2,30)]).sum()


class planck(SlikPlugin):

    def __init__(self, tau, output_file=None, lslice=slice(2,2509), sim=False, model='lcdm',
                 action='minimize', highl='custom', lowl='comm', simlow=False, derived=False, derived_cls=None):

        super(planck,self).__init__(**all_kw(locals()))

        def fix(*params):
            for p in params:
                self.cosmo[p] = self.cosmo[p].start

        assert model in ['lcdm','lcdmalens','Asns']
        self.cosmo = get_plugin('models.cosmology')(
            logA = param(3.108,0.03),
            ns = param(0.962,0.006),
            ombh2 = param(0.02221,0.0002),
            omch2 = param(0.1203,0.002),
            H0 = param(67,1),
            ALens = param(1,0.2),
            pivot_scalar = 0.05,
            mnu = 0.06
        )
        if 'alens' not in model: fix('ALens')
        if model=='Asns': fix('ombh2','omch2','H0')
        
        
        if simlow:
            self.cosmo.tau = param(0.07,0.01,min=0)
            self.simlow = clik(clik_file='simlow_MA_EE_2_32_2016_03_31.clik',auto_reject_errors=True)
        else:
            if isinstance(tau,tuple):
                self.cosmo.tau = param(tau[0],0.01,min=0,gaussian_prior=tau) 
            else:
                self.cosmo.tau = tau
            self.simlow = None

        self.get_cmb = get_plugin('models.camb')()
        self.get_cmb._camb.ignore_fatal_errors.value = True
        if derived: self.cambargs = {'redshifts':[0]}


        self.calPlanck=param(1,0.0025,gaussian_prior=(1,0.0025))

        class clikwrap(clik):
            def __call__(self,cl):
                return super(clikwrap,self).__call__({'cl_TT':cl*arange(cl.size)*(arange(cl.size)+1)/2/pi})

        if lslice.stop > 30:
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
        else:
            assert lslice.stop==30, "can't slice lowl likelihood"
            self.highl = None

        if lslice.start==2:
            if sim:
                assert lowl!='comm'
                self.lowl=lowl_approx(
                   mask_file=None if args.lowlfullsky else "commander_dx11d2_mask_temp_n0016_likelihood_v1.fits",
                   f=loadtxt("commander_dx11d2_mask_temp_n0016_likelihood_v1_f.dat") if lowl=='fl' else None,
                   fidcl=fidcl
                )
            else:
                if lowl=='comm':
                    self.lowl=clikwrap(clik_file='commander_rc2_v1.1_l2_29_B.clik')
                elif lowl in ['fl','fsky']:
                    self.lowl=lowl_approx(
                       mask_file="commander_dx11d2_mask_temp_n0016_likelihood_v1.fits",
                       maps=("COM_CMB_IQU-commander_1024_R2.02_halfmission-1.fits",
                             "COM_CMB_IQU-commander_1024_R2.02_halfmission-2.fits"),
                       bl=pyfits.open("COM_CMB_IQU-commander_1024_R2.02_full.fits")[2].data['INT_BEAM'],
                       f=loadtxt("commander_dx11d2_mask_temp_n0016_likelihood_v1_f.dat") if lowl=='fl' else None,
                       fidcl=fidcl
                    )
                else:
                    raise ValueError(lowl)

        else:
            assert lslice.start>=30, "can't slice lowl likelihood"
            self.lowl=None


        self.priors = get_plugin('likelihoods.priors')(self)

        #pick available covariance file closest to current lslice
        cov = sorted([(norm(array(map(int,re.search('([0-9]+)_([0-9]+)',f).groups()))-[lslice.start,lslice.stop]),f) 
                      for f in glob('covs/%s/*'%model)])
        if cov: cov = cov[0][1]
        else: 
            print("Warning: no covariance file found")
            cov = None

        if action=='minimize':
            self.sampler = Minimizer(self,whiten_cov=cov)
        else:
            extra_params = []
            if derived:
                extra_params += ['cosmo.'+k for k in [
                    'rdrag', 'rstar', 'age', 'kd', 'zdrag', 'thetaeq', 'zstar',
                    'thetarseq', 'zeq', 'thetad', 'thetastar', 'keq', 'DAstar',
                    'sigma8','theta','ommh2','clamp','s8omegamp25'
                ]]
            if derived_cls:
                if 'TT' in derived_cls: extra_params += [('clTT','(2510)d')]
                if 'pp' in derived_cls: extra_params += [('clpp','(501)d')]

            self.sampler = get_plugin('samplers.metropolis_hastings')(
                self,
                num_samples=1e6,
                print_level=2,
                mpi_comm_freq=50,
                output_extra_params=extra_params,
                output_file=output_file,
                proposal_cov=cov,
            )

    def add_derived(self):
        self.cosmo.update(self.get_cmb.result.get_derived_params())
        self.cosmo.sigma8 = self.get_cmb.result.get_sigma8()[0]
        self.cosmo.theta = self.get_cmb.result.cosmomc_theta()
        self.cosmo.ommh2 = self.cosmo.omch2 + self.cosmo.ombh2 + self.cosmo.mnu / 94.
        self.cosmo.clamp = exp(self.cosmo.logA - 2*self.cosmo.tau)/10
        self.cosmo.s8omegamp25 = self.cosmo.sigma8*(self.cosmo.ommh2/(self.cosmo.H0/100.)**2)**0.25
        self.clpp = self.get_cmb.result.get_lens_potential_cls(500)[:,0]

    def __call__(self):
        self.cosmo.As = exp(self.cosmo.logA)*1e-10
        if self.highl: 
            self.highl.A_Planck = self.highl.calPlanck = self.calPlanck
        if self.lowl: 
            self.lowl.A_planck = self.calPlanck

        p = {k:self.cosmo[k] for k in ['ns','As','ombh2','omch2','H0','tau','ALens'] if k in self.cosmo}
        p.update(self.get('cambargs',{}))
        p['lmax'] = 3000
        cmb = self.get_cmb(**p)
        self.clTT = cmb['cl_TT'][:2510]
        self.clTT[2:] *= (2*pi/arange(2,2510)/(arange(2,2510)+1))
        if self.derived: self.add_derived()

        return lsum(lambda: self.priors(self),
                    lambda: self.highl(self.clTT) if self.highl else 0,
                    lambda: self.lowl(self.clTT) if self.lowl else 0,
                    lambda: self.simlow(cmb) if self.simlow else 0)


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
            if args.debug: mspec_log(str((l,zip(self.sampled,y))))
            if args.progress:
                nfev[0]+=1
                with open('shared/progress','a') as f:
                    f.write('%.3f\n'%(0.001+nfev[0]/args.progress))
            return l

        x0=dot(inv(self.G),self.x0)+self.initial_scatter*normal(size=len(self.x0))
        if args.dryrun:
            yield dot(self.G,x0), dot(self.G,x0), None
        else:
            res = minimize(f,x0,method='Powell',options=dict(xtol=1e-4,ftol=1e-4,disp=False))
            yield dot(self.G,res.x), dot(self.G,x0), res




if __name__=='__main__':


    parser = argparse.ArgumentParser(prog='run_sim')
    parser.add_argument('--highl', default='custom', help='[custom|plik_like|plik]')
    parser.add_argument('--lowl', default='fl', help='[comm|fsky|fl]')
    parser.add_argument('--simlow', action='store_true', help='use simlow likelihod instead of tau prior')    
    parser.add_argument('--tau',  default="(0.07,0.02)", help='(mean,std) prior on tau, or just value to fix tau to')
    parser.add_argument('--sampletau',  action="store_true", help="sample tau from its prior for each sim")
    parser.add_argument('--lslices', help='e.g. [(2,800),(2,2500)]')
    parser.add_argument('--seeds', help='e.g. range(10)')
    parser.add_argument('--progress', metavar='MAXSTEPS', type=float, help='write progress to file')
    parser.add_argument('--real',action='store_true', help='do real data')
    parser.add_argument('--fid', default='fid_tau0.07.txt',help='fiducial Cls')
    parser.add_argument('--chain', action='store_true',help='run chain')
    parser.add_argument('--dryrun', action='store_true',help='only do one step of minimizer')
    parser.add_argument('--lowlfullsky', action='store_true',help='dont apply mask at lowl')
    parser.add_argument('--debug', action='store_true',help='print debug output')
    parser.add_argument('--derived', action='store_true',help='output derived parameters to chains')
    parser.add_argument('--derived_cls',help="any of pp,TT to output to chain")
    parser.add_argument('--model', default='lcdm',help='[lcdm|lcdmalens|Asns]')
    args = parser.parse_args()
    
    base_dir = osp.join(args.model,args.highl,args.lowl,
                        'simlow' if args.simlow else ('tau_'+'_'.join('%.3f'%x for x in atleast_1d(eval(args.tau)))))

    if args.fid:
        fidcl = loadtxt(args.fid)


    if args.chain:

        assert args.lslices is not None and len(eval(args.lslices))==1, "for chain must provide one single lslice"
        lslice = eval(args.lslices)[0]
        output_file = osp.join('shared/results/chains',base_dir,'lslice_%i_%i.chain'%lslice)
        try: os.makedirs(osp.dirname(output_file))
        except: pass
        p=Slik(planck(lslice=slice(*lslice),
                      sim=False,action='chain',
                      tau=eval(args.tau),
                      highl=args.highl,lowl=args.lowl,simlow=args.simlow,
                      derived=args.derived,
                      derived_cls=args.derived_cls.split(',') if args.derived_cls else None,
                      output_file=output_file,
                      model=args.model))
        for _ in p.sample(): pass

    else:

        def do_run(lslice, sim=False, seed=None):
            mspec_log('Doing: %s %s'%(lslice,'sim(%s)'%seed if seed is not None else ''))

            #prep result dir
            result_dir = osp.join('shared/results','sims' if sim else 'real',base_dir)
            if sim: result_dir = osp.join(result_dir,'sim_%s'%seed)
            try: os.makedirs(result_dir)
            except: pass
            result_file = osp.join(result_dir,'lslice_%i_%i'%lslice)
            if osp.exists(result_file):
                mspec_log('Skipped: %s %s'%(lslice,'sim(%s)'%seed if seed is not None else ''))
                return

            #set seed
            random.seed(seed)

            #prep tau
            tau = eval(args.tau)
            if args.sampletau:
                tau = normal(*tau), tau[1]
                print "tau prior: %f +/o %f"%tau

            #compute result
            s = Slik(planck(lslice=slice(*lslice),
                            sim=sim,
                            lowl=args.lowl,
                            highl=args.highl,
                            simlow=args.simlow,
                            tau=tau,
                            model=args.model))
            bf,x0,res = s.sample().next()

            #postprocessing
            s.params.cambargs = {'redshifts':[0]}
            lnl,p = s.evaluate(*bf)
            pp = dict(zip(s.get_sampled(),bf))
            pp['cosmo.tau'] = p.cosmo.tau
            pp.update({'cosmo.theta':100*p.get_cmb.result.cosmomc_theta(),
                       'cosmo.ommh2':pp['cosmo.omch2']+pp['cosmo.ombh2'],
                       'cosmo.sigma8':p.get_cmb.result.get_sigma8()[0],
                       'cosmo.clamp':exp(pp['cosmo.logA'])*exp(-2*pp['cosmo.tau'])/10,
                       'lnl':lnl,
                       'highl_chi2':2*p.highl(p.clTT),
                       'highl_dof':p.highl.bslice.stop - p.highl.bslice.start,
                       'clTT':p.clTT,
                       'res':res,
                       'inputtau':tau,
                       'x0':x0,
                       'seed':seed})
            pp['cosmo.s8omm1/4']=pp['cosmo.sigma8']*(pp['cosmo.ommh2']/(pp['cosmo.H0']/100.)**2)**0.25

            mspec_log('Got: %s'%str((lslice,pp)))

            #save result to file
            pickle.dump([((seed,lslice),pp)],
                        open(result_file,'w'),protocol=2)

            return ((seed,lslice),pp)



        if args.lslices is None:
            lslices = []
            for lsplit in range(100,2500,50)+[2509]:
                if lsplit<1700: lslices.append((lsplit,2509))
                for lmin in (2,30):
                    if lsplit>=650: lslices.append((lmin,lsplit))
        else:
            lslices = eval(args.lslices)

        mspec_log('Doing lslices (%i): %s'%(len(lslices),lslices),rootlog=True)


        if args.real:

            results = mpi_map(do_run,lslices)

        else:

            assert args.seeds is not None, "must provide seeds"

            mpi_map(lambda seed: [do_run(lslice,sim=True,seed=seed) for lslice in lslices], eval(args.seeds))
