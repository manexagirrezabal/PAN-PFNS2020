
import numpy as np

#Embeddings from https://www.spinningbytes.com/resources/wordembeddings/

LANG="es"
f=open("embeddings/embedding_file_"+LANG)
vsizestr,nembsstr = f.readline().strip().split()
lines = [line.strip().split() for line in f]
f.close()

nembs = int(nembsstr)
vsize= int(vsizestr)

print ("loading")
wdict = {}
for indl,line in enumerate(lines):
    print ("Loading word embeddings ", str(indl)+"/"+str(vsize),end="\r")
    word = line[0]
    emb = line[1:]
    if len(emb) == nembs:
        wdict[word] = np.array(emb,dtype=np.float)


wembs = (wdict,nembs,vsize)
