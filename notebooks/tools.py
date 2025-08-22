import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pyfaidx import Fasta

shared_dir = os.path.expanduser('~/Library/Mobile Documents/com~apple~CloudDocs/What is SBS5')
A,C,G,T = 'A','C','G','T'

class COSMIC():
    """ This class introduces cosmic order and 
        cosmic-like plots for signatures """
    
    def __init__(self,loc=os.path.join(shared_dir, 'misc_files/COSMIC_v3.3.1_SBS_GRCh38.txt')):
        
        self.order = ['A[C>A]A', 'A[C>A]C', 'A[C>A]G', 'A[C>A]T',
                      'C[C>A]A', 'C[C>A]C', 'C[C>A]G', 'C[C>A]T',
                      'G[C>A]A', 'G[C>A]C', 'G[C>A]G', 'G[C>A]T',
                      'T[C>A]A', 'T[C>A]C', 'T[C>A]G', 'T[C>A]T',
                      'A[C>G]A', 'A[C>G]C', 'A[C>G]G', 'A[C>G]T',
                      'C[C>G]A', 'C[C>G]C', 'C[C>G]G', 'C[C>G]T',
                      'G[C>G]A', 'G[C>G]C', 'G[C>G]G', 'G[C>G]T',
                      'T[C>G]A', 'T[C>G]C', 'T[C>G]G', 'T[C>G]T',
                      'A[C>T]A', 'A[C>T]C', 'A[C>T]G', 'A[C>T]T',
                      'C[C>T]A', 'C[C>T]C', 'C[C>T]G', 'C[C>T]T',
                      'G[C>T]A', 'G[C>T]C', 'G[C>T]G', 'G[C>T]T',
                      'T[C>T]A', 'T[C>T]C', 'T[C>T]G', 'T[C>T]T',
                      'A[T>A]A', 'A[T>A]C', 'A[T>A]G', 'A[T>A]T',
                      'C[T>A]A', 'C[T>A]C', 'C[T>A]G', 'C[T>A]T',
                      'G[T>A]A', 'G[T>A]C', 'G[T>A]G', 'G[T>A]T',
                      'T[T>A]A', 'T[T>A]C', 'T[T>A]G', 'T[T>A]T',
                      'A[T>C]A', 'A[T>C]C', 'A[T>C]G', 'A[T>C]T',
                      'C[T>C]A', 'C[T>C]C', 'C[T>C]G', 'C[T>C]T',
                      'G[T>C]A', 'G[T>C]C', 'G[T>C]G', 'G[T>C]T',
                      'T[T>C]A', 'T[T>C]C', 'T[T>C]G', 'T[T>C]T',
                      'A[T>G]A', 'A[T>G]C', 'A[T>G]G', 'A[T>G]T',
                      'C[T>G]A', 'C[T>G]C', 'C[T>G]G', 'C[T>G]T',
                      'G[T>G]A', 'G[T>G]C', 'G[T>G]G', 'G[T>G]T',
                      'T[T>G]A', 'T[T>G]C', 'T[T>G]G', 'T[T>G]T']
        self.contexts = [t[0]+t[2]+t[6] for t in self.order]
        self.colors = [[3/256,189/256,239/256],
                       [1/256,1/256,1/256],
                       [228/256,41/256,38/256],
                       [203/256,202/256,202/256],
                       [162/256,207/256,99/256],
                       [236/256,199/256,197/256]]
        
        bad_order = pd.read_table(loc)
        good_order = np.array(self.order).argsort().argsort()
        self.cosmic = pd.DataFrame(bad_order.to_numpy()[good_order], columns=bad_order.columns)
        assert (self.cosmic.Type==self.order).all()
        A,C,G,T = 'A','C','G','T'
        self.RC = {A: T, C: G, G: C, T: A}
        self.Alt = {A: [C,G,T], C: [A,G,T], G: [A,C,T], T: [A,C,G]}

    def revContext(self,context):
        if len(context) < 3:
            return np.nan
        return self.RC[context[2]] + self.RC[context[1]] + self.RC[context[0]]

    def revType(self,typ):
        return self.RC[typ[6]]+typ[1]+self.RC[typ[2]]+typ[3]+self.RC[typ[4]]+typ[5]+self.RC[typ[0]]
        
    def prepareFold(self,types_column):
        generalTypes = set(types_column)
        self.foldDict = dict(zip(generalTypes, generalTypes))
        for typ in generalTypes:
            if typ[2] in ['A','G']:
                self.foldDict[typ] = self.revType(typ)
        
    def fold(self,types_column):
        """ Map 192 to 96 mutation types
            A[C>T]G is A[C>T]G (COSMIC red)
            but so is  C[G>A]T """ 
        self.prepareFold(types_column)
        return types_column.map(self.foldDict)
        
    def plot(self,proba,
             save=None,
             title=None,
             show_letters=False,
             figsize=(6,2),
             width=1):
        """ COSMIC-like plot """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        delta = len(self.order)//len(self.colors)
        x = np.arange(delta)   
        for i in range(6):
            ax.bar(x+i*delta,proba[i*delta:(i+1)*delta],
                   color=self.colors[i],width=width)
        ax.set_xlim(-width/2,len(self.order)-width/2)
        ax.set_ylim(ymin=0)
        ax.set_xticks(range(len(self.order)))
        if show_letters: ax.set_xticklabels(self.contexts,rotation=90)
        else: ax.set_xticklabels(['' for _ in self.contexts],rotation=90)
        ax.set_xlabel('Mutation types')
        ax.set_ylabel('Distribution')
        ax.legend([],title=title,frameon=False,loc=1)
        plt.tight_layout()
        if save is not None: plt.savefig(save)
        plt.show()
        
class EM():
    """ This class estimates global signature proportions
        (theta) using Expectation-Maximization """
    
    def __init__(self,df,cosmic,beta=0,
                 cutoff=None,
                 choice=None,
                 cosmicType='Type',
                 rescale=None):
        self.order = cosmic.order
        self.cosmicType = cosmicType
        columns = cosmic.cosmic.columns
        if cutoff is not None: self.cosmic = cosmic.cosmic[signs[:cutoff+1]]       
        elif choice is not None: self.cosmic = cosmic.cosmic[[self.cosmicType]+choice]       
        else: self.cosmic = cosmic.cosmic
        if rescale is not None:
            ps = cosmic.cosmic[columns[1:]].values
            psRescaled = (ps.T*rescale)
            zs = psRescaled.sum(axis=1)
            (psRescaled.T/zs).shape
            cosmic.cosmic[columns[1:]] = psRescaled.T/zs
        
        self.dim = len(self.cosmic.columns)-1
        self.theta = np.ones(self.dim)/self.dim
        self.x = df.groupby([self.cosmicType]).size().reindex(self.order, fill_value=0)
        # regularization: Dirichlet prior with strength beta
        self.beta = beta 
        
    def expect(self,typ):
        """ Compute membership probabilities
            given theta """
        likelihood = self.cosmic.loc[self.cosmic.Type==typ].values[0,1:]
        membership = likelihood*self.theta
        return membership/membership.sum()
    
    def maximize(self):
        """ Update theta
            given membership probabilities """
        theta = np.zeros(self.dim,dtype=float)
        for typ in self.order:
            theta = theta + self.expect(typ)*self.x[typ]
        # regularization: Dirichlet prior with strength beta
        theta -= self.beta
        # component anihilation 
        theta[theta<0] = 0
        self.theta = theta/theta.sum()

    def div(self,p1,p2):
        """ 1 - cos similarity """ 
        z = np.sqrt(np.sum(p1*p1)*np.sum(p2*p2))
        return 1 - np.sum(p1*p2)/z
    
    def infer(self,epsilon=1e-10):
        """ Iterative E-M """
        while True: 
            old_theta = self.theta
            self.maximize()
            if self.div(old_theta,self.theta) < epsilon: break
        
    def observed(self):
        """ Returns observed signature """
        ps = np.zeros(len(self.order))
        for i,typ in enumerate(self.order):
            ps[i] += self.x[typ]
        return ps/sum(ps)
    
    def reconstructed(self):
        """ Returns reconstructed signature """
        ps = (self.cosmic.values[:,1:].astype(float)*self.theta).sum(axis=1)
        return ps/sum(ps)
    
    def plot(self,
             figsize=(6,2),
             color=None,
             title=None,
             save=None,
             first=25):
        """ Plot global signature proportions """
        shift = 0.5
        width = 1
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.bar(range(self.dim),self.theta[np.argsort(-self.theta)],
               color=color,width=width)
        ax.set_xlim(0-shift,first-shift)
        ax.set_xticks(range(first))
        ax.set_xticklabels(self.cosmic.columns[1:][np.argsort(-self.theta)][:first],
                           rotation=90)
        ax.set_ylabel('Distribution')
        ax.legend([],frameon=False,loc=1,title=title)
        plt.tight_layout()
        if save is not None: plt.savefig(save)
        plt.show()
        
    def signatureProba(self,signature,df):
        """ Estimate probability that mutation
            follow from given signature """
        assert signature in self.cosmic.columns
        index = np.where(self.cosmic.columns==signature)[0][0]-1
        ps = self.cosmic.values[:,1:]*self.theta # Bayes rule
        probaDict = dict(zip(self.cosmic.Type,(ps.T/ps.sum(axis=1))[index]))
        return df[self.cosmicType].map(probaDict)

class Reference():
    """ This class uses genomic reference """
    
    def __init__(self,ref=19,dirname=None):
        assert (ref==19) or (ref==38)
        if dirname is not None: self.dirname = dirname
        else: dirname = os.path.expanduser('~/Library/Mobile Documents/com~apple~CloudDocs/What is SBS5/data/')
        
        self.hg = Fasta(dirname + 'fasta/hg{}.fa'.format(ref))
        self.hgRefChr = {}
        for i in list(range(1,22+1)) + ['X', 'Y']:
            key = 'chr{}'.format(i)
            self.hgRefChr.update({str(i): key, key: key})
    
    def findRef(self,Chr,Pos):
        return self.hg[self.hgRefChr[Chr]][Pos-1].seq.upper()

    def findContext(self,Chr,Pos):
        return self.hg[self.hgRefChr[Chr]][Pos-2:Pos+1].seq.upper()

    def check(self,df,chr_col='Chr',pos_col='Pos',pos_shift=0,ref_col='Ref'):
        df['Chr'] = df[chr_col].astype(str).str.replace('chr','')
        df['Pos'] = df[pos_col]+pos_shift
        df['Ref-check'] = df.apply(lambda x: self.findRef(x['Chr'], x['Pos']), axis=1)
        self.errors = df['Ref-check']!=df[ref_col]
        return sum(df['Ref-check']!=df[ref_col])/len(df)
    
    def assembleType(self,df,
                     context_col='Context',alt_col='Alt',
                     ref_col=None,
                     cosmic=None):
        if cosmic is None: cosmic = COSMIC()
        if ref_col is None:
            df['GeneralType'] = df[context_col].str[0]+'['+df[context_col].str[1]+'>'+df[alt_col]+']'+df[context_col].str[2]
        else:
            df['GeneralType'] = df[context_col].str[0]+'['+df[ref_col]+'>'+df[alt_col]+']'+df[context_col].str[2]
        return cosmic.fold(df['GeneralType'])
    
def get_trinucleotide(row, chrom_seqs):
    """
    Extracts the trinucleotide context of a mutation from a fasta file  
    """
    chrom = row["Chr"]
    start = row["Pos"]-1
    end = row["Pos"]
    ref = row["Ref"]

    ref_context = chrom_seqs[chrom][start-1:end+1]
    
    if ref!=ref_context[1]:
        pass
        #sys.stderr.write("Wrong reference fasta. Ref in mutation type doesn't match the ref fasta")

    return ref_context

def keep_snps(df):
    """
    Removes indels from a mutation dataframe, 
    i.e., sites at which len(ref)>1 or len(alt)>1
    """
    return df.query("Ref.str.len() == 1 and Alt.str.len() == 1").reset_index(drop=True)

def annotate_general_type(df):
    """
    Returns general class of mutation, without folding
    """
    return df.Context.str[0]+'['+df.Context.str[1]+'>'+df.Alt+']'+df.Context.str[2]

class EM_with_clock():
    """ Estimates clock-like signature proportions
        (theta) using Expectation-Maximization """

    def __init__(self, df, cosmic, cutoff=None, choice=None,
                 cosmicType='Type', ageLabel='Age', speed=0.04):
        self.order = cosmic.order
        self.ageLabel = ageLabel
        if cutoff is not None:
            self.cosmic = cosmic.cosmic[cosmic.columns[:cutoff+1]]
        elif choice is not None:
            self.cosmic = cosmic.cosmic[[self.cosmicType]+choice]
        else:
            self.cosmic = cosmic.cosmic
        self.dim = len(self.cosmic.columns) - 1
        self.theta = np.ones((2,self.dim),dtype=float) / self.dim
        
        data = pd.DataFrame(df.groupby([ageLabel, cosmicType]).size()).reset_index()
        t = data[self.ageLabel].values
        x = data[cosmicType].values
        self.count = data[0].values.astype(float)
        self.x_given_s = self.cosmic.set_index('Type').loc[x].values.astype(float)
        self.p0_given_t = (1 / (speed * t + 1)).astype(float)
        self.p1_given_t = (1 - self.p0_given_t).astype(float)
                
    def infer(self):
        """ Iteration of expectation and maximization """
        while True:
            old_theta = self.theta.copy()
            
            """ Compute membership probabilities given theta """
            membership0 = (self.x_given_s * self.theta[0]).T * self.p0_given_t
            membership1 = (self.x_given_s * self.theta[1]).T * self.p1_given_t
            Z = membership0.sum(axis=0) + membership1.sum(axis=0)
            probabilities = np.array([membership0,membership1])/Z

            """ Update theta given membership probabilities """
            theta = (probabilities * self.count).sum(axis=-1).T
            self.theta = (theta/theta.sum(axis=0)).T
            
            if np.allclose(old_theta, self.theta): break
