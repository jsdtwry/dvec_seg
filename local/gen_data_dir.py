import sys
import os

uttlst = file('lst/thu_ev.lst').readlines()

for i in uttlst:
	each = i[:-1].split(' ')
	os.makedirs('data/'+each[0])
	fout = file('data/'+each[0]+'/wav.scp','w')
	fout.write(i)

