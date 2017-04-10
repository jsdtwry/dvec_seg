import sys
import os

flist = file('lst/thu_ev.lst').readlines()

for i in flist:
	each = i[:-1].split(' ')
	#os.system('sh fea_extract.sh data/'+each[0]+'/')
	os.system('python2.7 local/cal_bic_curves.py data/'+each[0]+'/ 1.0 0.3')
	os.system('python2.7 local/cal_dvec_curves.py data/'+each[0]+'/ 0.1 0.01')
	#print 'extract score feature of', each

print 'extract score feature finish!'
