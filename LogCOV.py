'''
author: limengjie
Paper: "When Kernel Methods meet Feature Learning: Log-Covariance Network for Action Recognition from Skeletal Data"
'''
import numpy as np
import json
import pickle as pk
import os
import math
import random

dataPath="./MSRAction3DSkeleton(20joints)"
def loaddata(datapath):
	#see function(1)
	#every 20rows in *.txt is for one frame
	#load data from each frame of each video
	row=[]
	content=[]
	with open(datapath,'rb') as f:
		for eachline in f:
			for ele in eachline.strip().split():
				row.append(float(ele))
			content.append(row)
			row=[]
	content=np.asarray(content)
	content=content[:,:3]#get rid of the element "score"
	content=content.reshape(content.shape[0]//20,20,3)#[frames,20joints,joints_coordinates]
	content_mean=[]
	for cont in content:
		temp=cont-cont[6,:]#set joint7 as the root point, as it refered in paper that "Typically the hip center is adopted as the root"
		content_mean.append(temp.tolist())
	content_mean=np.asarray(content_mean)
	content_mean=content_mean.reshape(content_mean.shape[0],-1)#[frames, 20joints*joints_coordinates]
	pk_new=content_mean.T#3J*T  #[20joints*joints_coordinates, frames]
	
	return pk_new

def LogCOV(Pa):
	#construct Xa  see function(2)
	Pa=np.asarray(Pa)
	timestep=Pa.shape[1]
	identityI=np.identity(timestep)
	ones1=np.ones([timestep,timestep])
	tmp=((1.0/timestep)*identityI-ones1)
	Xa=(1.0/(timestep-1))*np.dot(np.dot(Pa,tmp),Pa.T)

	#compute eigendecomposition  see function(4)
	u,sigma,_=np.linalg.svd(Xa)
	sigma=sigma+10**(-4)
	sigma=np.log(sigma)
	identity_sigma=sigma*np.identity(Xa.shape[0])
	log_Xa=np.dot(np.dot(u,identity_sigma),u.T)
	'''
	#-------------------------------------------
	#this part is as referred that"Precisely, we define v a to be the vectorization of all diagonal and lower-diagonal entries"
	Va_diagonal=np.diagonal(log_Xa)
	Va_lower_diagonal=np.diagonal(log_Xa,-1)#offset=-1
	assert Va_diagonal.ndim==1
	assert Va_lower_diagonal.ndim==1
	Va=Va_diagonal.tolist()
	Va.extend(Va_lower_diagonal.tolist())
	#-------------------------------------------
	'''
	#-------------------------------------------
	#this part is to use lower triangular as feature
	indices=np.tril_indices(log_Xa.shape[0])
	Va=log_Xa[indices]
	#-------------------------------------------
	return Va

#main
def main():
	fea=[]
	label=[]
	for video_tuple in os.walk(dataPath):
		for video in video_tuple[2]:
			feature=loaddata(os.path.join(dataPath,video))
			classname=int(video[1:3])
			fea.append(LogCOV(feature))
			label.append(classname-1)
	count=len(fea)
	print("totally %d videos in MSR3D" % count)
		
	#train validation test split
	train_num=int(0.7*count)
	val_num=int(0.1*count)
	#test_num=count-train_num-val_num
	fea_list=range(count)
	random.shuffle(fea_list)
	train_list=fea_list[:train_num]
	val_list=fea_list[train_num:train_num+val_num]
	test_list=fea_list[train_num+val_num:]
	train_fea=[]
	train_label=[]
	for index in train_list:
		train_fea.append(fea[index])
		train_label.append(label[index])
	val_fea=[]
	val_label=[]
	for index in val_list:
		val_fea.append(fea[index])
		val_label.append(label[index])
	test_fea=[]
	test_label=[]
	for index in test_list:
		test_fea.append(fea[index])
		test_label.append(label[index])
		
	MSR3D={"train_fea":train_fea,"train_label":train_label,"val_fea":val_fea,"val_label":val_label,"test_fea":test_fea,"test_label":test_label}
	with open("./feature_label_MSR3D_lower_triangular.pkl",'wb') as f1:
		pk.dump(MSR3D,f1)

	print len(train_label)
	print len(val_label)
	print len(test_label)

if __name__=='__main__':
	main()












