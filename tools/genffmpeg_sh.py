import os
import sys
from os.path import join
import os.path

def listFiles(dirpath,suffix):
        listlines=[]
        files = os.listdir(dirpath)
        for eachfile in files:
                curfile = dirpath + os.sep + eachfile
                if os.path.isdir(curfile):
                        for mid in listFiles(curfile,suffix):
                                listlines.append(mid)
                elif eachfile.endswith(suffix):
                        listlines.append([dirpath,eachfile])
        return listlines
'''
listf = listFiles('THU_EB_1062_201612/Database','mp3')
#listf = listFiles('thu_ev','mp3')
fout = file('format.sh','w')

for i in listf:
	fout.write('./ffmpeg -i '+i[0]+'/'+i[1]+' -ar 8000 thu_ev_wav/'+i[1][:-3]+'wav\n')
'''

listf = listFiles('thu_ev_wav','wav')
fout = file('thu_ev.lst','w')

for i in listf:
	fout.write(i[1][:-4]+' /nfs/user/wangrenyu/data/thu_ev_xmx_1/thu_ev_wav/'+i[1]+'\n')



