import sys,os,logging
import pandas as pd
import numpy as np
import glob
from importlib import reload
import statsmodels.api as sm
from statsmodels.stats.nonparametric import rank_compare_2indep
import re
from itertools import compress
import itertools
sys.path.append("/ix/schan/ukbb")
from ukbb_library import *
from IPython.display import display
pd.set_option('display.max_columns', None)


def exportLR(LR, SSS, paraEst, pcol, outfile_suffix, l, suffix, output_dir, instance, subscript=None):
    if paraEst in ['bootstrap','permutation_ntrials','permutation_exact','groupMean_bootstrapping','groupMean_permutation']:
      paraEst = "_"+paraEst
    else:
      paraEst=""
    if LR.empty:
      print(l, " has no usable variable")
      return(None)
    I = np.where(LR.loc[caseCol,pcol] <= 1.0)[0]
    if(len(I)==0):
      print(l, " has no significant result")
      return(None)
    padj = (ss.multitest.fdrcorrection(LR.loc[caseCol,pcol]))[1]
    padjBonf = (ss.multitest.multipletests(LR.loc[caseCol,pcol], method='bonferroni'))[1]
    target_I = int(np.where(LR.index==caseCol)[0])
    b = pd.DataFrame([LR.iloc[0,np.sort(np.concatenate([I*5]))].values, LR.iloc[target_I,np.sort(np.concatenate([I*5]))].values, LR.iloc[target_I,np.sort(np.concatenate([I*5+2]))].values, padj[I], padjBonf[I], LR.iloc[target_I,np.sort(np.concatenate([I*5+3]))].values, LR.iloc[target_I,np.sort(np.concatenate([I*5+4]))].values, LR.iloc[1,np.sort(np.concatenate([I*5]))].values, LR.iloc[2,np.sort(np.concatenate([I*5]))].values]).T
    b.columns = ['DF','Coefficient','P-value','Padj','Padj_Bonf','CI_025','CI_975','Sample_Count','Sample_Count_Case']
    b['DF'+instance] = b['DF'].copy()
    b['DF'] = [re.sub("^(.+)-.+$","\\1",i) for i in b['DF'] ]
    if b['DF'][0].isnumeric():
      b['DF'] = b['DF'].astype('int')
    SSS.index = b.index
    b = pd.concat([b, SSS], axis=1)
    temp = b
    temp = temp.sort_values('P-value')
    temp.to_csv(output_dir+'/sig2_DFs-Case_LR'+outfile_suffix+'__'+l+"_"+suffix+paraEst+subscript+'.csv',sep=",",header=True,index=False)

def permutationOLSregression(X2, formula, dvmax, resT, caseCol, pcol, pvalMethod, ntrials):
  if ntrials=='exact':
    print('exact permutation')
    combinations = list(itertools.combinations(list(range(0, X2.shape[0])), (X2[caseCol]==1).sum()))
    params = pd.DataFrame([],index=X2.columns.drop(['IDF']),columns=list(range(0,len(combinations))))
    for trial in range(len(combinations)):
      data_sample = X2.copy()
      data_sample[caseCol] = 0
#      data_sample[caseCol].iloc[sample(range(X2.shape[0]), k=X2[caseCol].sum())] = 1
      data_sample[caseCol].iloc[np.array(combinations[trial])] = 1
      temp = ols(formula, data_sample).fit()
      params.loc[caseCol,trial] = temp.params[caseCol]
  elif isinstance(ntrials, (int, float)):
    print('ntrial permutation')
    params = pd.DataFrame([],index=X2.columns.drop(['IDF']),columns=list(range(0,ntrials)))
    for trial in range(ntrials):
      data_sample = X2.copy()
      data_sample[caseCol] = 0
      data_sample[caseCol].iloc[sample(range(X2.shape[0]), k=X2[caseCol].sum())] = 1
      temp = glm(formula, data_sample).fit()
      params.loc[caseCol,trial] = temp.params[caseCol]
  params = params[params.isna().sum(axis=1)!=ntrials]
  # confidence interval
  sorted_perm_stats = np.sort(np.array(params))
  sorted_perm_stats = sorted_perm_stats - resT.loc[caseCol,'Coef.']
  lower_bound = np.percentile(sorted_perm_stats, 2.5)
  upper_bound = np.percentile(sorted_perm_stats, 97.5)
  resT.loc[caseCol,'[0.025'] = lower_bound
  resT.loc[caseCol,'0.975]'] = upper_bound#  confidence_level = 0.95
  # pvalue
  shapiroP = stats.shapiro(params.loc[caseCol,:])[1]
  print('Shapiro test:'); print(shapiroP)
  if pvalMethod=='ztest':
    pval = ztest(params.loc[caseCol,:], value=resT.loc[caseCol,'Coef.'])
    resT.loc[caseCol,pcol] = pval[1]
  elif pvalMethod=='count-based': 
    gt = (params.loc[caseCol,:] >resT.loc[caseCol,'Coef.']).sum()
    lt = (params.loc[caseCol,:] <resT.loc[caseCol,'Coef.']).sum()
    resT.loc[caseCol,pcol] = np.min([gt,lt]) / (gt+lt)
  return(resT)

infile = sys.argv[1] # e.g. adjusted_data_LBRvolume_nativeSpace.csv
adjV = sys.argv[2].lower()=='true' # False
os.chdir('/ix/schan/ukbb/PH/mouseMRI/04Sep2024_MRI/2024-7-25/input_nii_files/9s')
l = infile.replace('adjusted_data_', '').replace('.csv', '')
M = pd.read_csv(infile,sep=',',index_col=0)
M.rename({'Age (weeks)':'Age','Group':'Case'},axis=1,inplace=True)
M.columns = M.columns.str.replace("-","_")
if adjV:
    Ys = [col for col in M.columns if col.startswith("adj_")]
    COVs = [['Age']]  # Only adjust for Age
    adj_label = "adj_"
else:
    Ys = [col for col in M.columns if not col.startswith("adj_") and col not in [ 'Ear Tag', 'rack', 'Case', 'Sex', 'DOB', 'Age', 'Weight (g)', 'RVSP', 'RV (mg)', 'LV+S (mg)', 'RV/LV+S', 'RV/MASS', 'Whole_Brain_Volume']]
    COVs = [['Age', 'Whole_Brain_Volume'],['Whole_Brain_Volume']]  # Adjust for Age & Whole Brain Volume
    adj_label = ""

# linear regression with age and Whole_Brain_Volume as a covariate
caseCol='Case'
LRtype='DFs-Case_LR'
pcol = 'P>|z|'
comparisons = [['IL6','WT'],['IL6_NCOA7_KO','WT'],['IL6_NCOA7_KO','IL6']]

for cov in COVs:
  for com in comparisons:
    X = M[M['Case'].isin(com)]
    X['Case'] = (X['Case']==com[0]).astype(int)
    subscript = f"_{adj_label}{'-'.join(com)}_{'-'.join(cov)}"
    LR1 = pd.DataFrame(); LR2 = pd.DataFrame(); LR3 = pd.DataFrame(); LR4 = pd.DataFrame(); LR5 = pd.DataFrame(); LR6 = pd.DataFrame()
    SSS = pd.DataFrame([])
    for dv in Ys:  
      LoR = False
      dv = str(dv)
      if not dv in X.columns:
        print(dv," does not exist")
      else:
        print(dv)
      X2 = X[[dv]+['Case']+cov]
      X2.rename({dv:'IDF'},axis=1,inplace=True)
      formula = "IDF" + " ~ " + ' + '.join(X2.drop(['IDF'],axis=1).columns.tolist())
      resT = runGLM_formula(X2, formula, dv, printResult=True)
      # stats
      sub1 = pd.DataFrame(X2[X2[caseCol]==1],columns=['IDF'])
      sub3 = pd.DataFrame(X2[X2[caseCol]==0],columns=['IDF'])
      mean1 = sub1.mean(axis=0); std1=sub1.std(axis=0); min1=sub1.min(axis=0); q5_1=sub1.quantile(0.05, axis=0); q25_1=sub1.quantile(0.25, axis=0); med1=sub1.quantile(0.5, axis=0); q75_1=sub1.quantile(0.75, axis=0); q95_1=sub1.quantile(0.95, axis=0); max1=sub1.max(axis=0)
      mean3 = sub3.mean(axis=0); std3=sub3.std(axis=0); min3=sub3.min(axis=0); q5_3=sub3.quantile(0.05, axis=0); q25_3=sub3.quantile(0.25, axis=0); med3=sub3.quantile(0.5, axis=0); q75_3=sub3.quantile(0.75, axis=0); q95_3=sub3.quantile(0.95, axis=0); max3=sub3.max(axis=0)
      de = mean1 - mean3
      de_percent = de / mean3 *100
      ms1 = np.round(mean1,2).astype(str) + " \xb1 " + np.round(std1,2).astype(str)
      ms3 = np.round(mean3,2).astype(str) + " \xb1 " + np.round(std3,2).astype(str)
      sss = pd.DataFrame([ms1, ms3, de, de_percent, min1,q5_1,q25_1,med1,q75_1,q95_1,max1,min3,q5_3,q25_3,med3,q75_3,q95_3,max3],index=['Case_mean[std]','Control_mean[std]','Case-Control','Percent_Case-Control','Case_min','Case_q5','Case_q25','Case_q50','Case_q75','Case_q95','Case_max','Control_min','Control_q5','Control_q25','Control_q50','Control_q75','Control_q95','Control_max']).astype(str).T
      SSS = pd.concat([SSS,sss],axis=0)
      # merge
      LR3 = pd.concat([LR3, permutationOLSregression(X2, formula, dv, resT.copy(), caseCol, pcol=pcol, pvalMethod='count-based', ntrials=1000)], axis=1)# 
    exportLR(LR3, SSS, "permutation", pcol, "", l, 'PAH_instance2', '.', '-2.0',subscript)


