from itertools import product
import os
import sys
import pandas
import numpy
import pickle
def sample_formulation(seq,k,kmers):
	for i in range(len(kmers)):
		kmers[i] = ''.join(kmers[i])
	kmer_in_seq = []
	for i in range(len(seq)-k+1):
		kmer = seq[i:i+k]
		kmer_in_seq.append(kmer)
	count = []
	for i in kmers:
		count.append(kmer_in_seq.count(i))
	#feature_vector = count/len(kmer_in_seq)
	feature_vector = [str(c/len(kmer_in_seq)) for c in count]
	return feature_vector
k=int(sys.argv[2])
base = 'ATCG'
# ATCG全排列
kmers = list(product(base, repeat=k))
filename = sys.argv[1]

with open(filename) as f:
	lines = f.readlines()
seq = []
for i in range(len(lines)):
	if i % 2 == 1: #分类问题中需注释掉
		seq.append(lines[i].strip().upper())
		print("i="+str(i))
feature_vectors = []
num = 0
for j in seq:
	feature_vectors.append(sample_formulation(j,k,kmers))
	num = num +1
	print(num)
print(type(feature_vectors))
filename = filename.split('.')
feature_vectors = pandas.DataFrame(feature_vectors)
feature_vectors.to_csv(filename[0]+"_"+str(k)+'mer'+'.txt', index=False, header = False,encoding='gbk',float_format='%.4f',sep = " ")
print(feature_vectors)
numpy.savetxt(str(len(feature_vectors[0]))+filename[0]+'.txt',feature_vectors)
#pickle.dump(feature_vectors,str(len(feature_vectors[0]))+filename[0]+'.txt')
#feature_vector = sample_formulation(test,2,kmers)
#第一个命令行参数是fasta文件的filename，第二个参数是k
# kmer_1.py pos.fa 5    放在同一层文件夹下