import pdb
import os
import numpy as np 
import scipy.stats as stats
from sklearn import metrics
import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

def get_loss(pred, labels, type='new_mse'):
	bias_ = 1e-6
	if type == 'new_mse':
		mean_pred = torch.mean(pred)
		mean_labels = torch.mean(labels)
		normalized_pred = (pred - mean_pred) / torch.sqrt(torch.var(pred)+bias_)
		normalized_labe = (labels - mean_labels) / torch.sqrt(torch.var(labels)+bias_)
		loss_new_mse = torch.mean((normalized_pred - normalized_labe)**2, dim=0)
		return loss_new_mse * 100.0
	elif type == 'pearson':
		mean_pred = torch.mean(pred)
		mean_labels = torch.mean(labels)
		loss_pearson =  torch.tensor(1.0).cuda() - torch.sum((pred - mean_pred) * (labels - mean_labels)) \
			/ (torch.sqrt(torch.sum((pred - mean_pred)**2) * torch.sum((labels - mean_labels)**2) + bias_))
		return loss_pearson * 100.0
	elif type == 'mse':
		loss_mse = torch.mean((pred - labels)**2, dim=0)
		return loss_mse 
	elif type == 'huber':
		crit = torch.nn.SmoothL1Loss()
		return crit(pred, labels)
	else:
		return None

class ASS_JRG(nn.Module):
	def __init__(self, whole_size=400, patch_size=19*50, seg_num=10, joint_num=17, mode='base'):
		super(ASS_JRG,self).__init__()
		self.whole_size = whole_size
		self.patch_size = patch_size
		self.seg_num = seg_num
		self.joint_num = joint_num
		self.module_num = 12
		#if mode == 'identity':
		#	self.module_num = 3
		self.dropout_rate = 0.1 #0.2
		self.mode = mode
		self.hidden1 = 256 
		self.hidden2 = 256
		self.hidden3 = 32
		# Joint Relation Graphs
		if self.joint_num == 17:
			a_file = './mat_a.npy'
			if not os.path.isfile(a_file):
				a_file = './mat_a.npy'
			a = np.load(a_file).astype(float) + np.identity(self.joint_num)
		else:
			a = np.array([[1,1,1,0],
					[1,1,0,1],
					[1,0,1,0],
		 			[0,1,0,1]])
		a2 = np.matmul(a,a)*(1-a)
		for i in range(self.joint_num):
			for j in range(self.joint_num):
				if a2[i][j] > 1:
					a2[i][j] = 1
		a3 = np.matmul(a2,a)*(1-a)*(1-a2)
		for i in range(self.joint_num):
			for j in range(self.joint_num):
				if a3[i][j] > 1:
					a3[i][j] = 1
		a4 = np.matmul(a3,a)*(1-a)*(1-a3)
		for i in range(self.joint_num):
			for j in range(self.joint_num):
				if a4[i][j] > 1:
					a4[i][j] = 1
		self.a = a
		self.a2 = a2
		self.a3 = a3
		self.a4 = a4

		self.spatial_mat1 = nn.Linear(self.joint_num, self.joint_num, bias=False).cuda()
		self.spatial_mat2 = nn.Linear(self.joint_num, self.joint_num, bias=False).cuda()
		self.spatial_mat3 = nn.Linear(self.joint_num, self.joint_num, bias=False).cuda()
		self.spatial_mat4 = nn.Linear(self.joint_num, self.joint_num, bias=False).cuda()

		self.temporal_mat1 = nn.Linear(self.joint_num, self.joint_num, bias=False).cuda()
		self.temporal_mat2 = nn.Linear(self.joint_num, self.joint_num, bias=False).cuda()
		self.temporal_mat3 = nn.Linear(self.joint_num, self.joint_num, bias=False).cuda()
		self.temporal_mat4 = nn.Linear(self.joint_num, self.joint_num, bias=False).cuda()
		# Aggregators from Joint Difference Module
		self.spatial_JCW1 = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.temporal_JCW1 = nn.Linear(self.joint_num, 1, bias=False).cuda()
		
		self.spatial_JCW2 = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.temporal_JCW2 = nn.Linear(self.joint_num, 1, bias=False).cuda()
		
		self.spatial_JCW3 = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.temporal_JCW3 = nn.Linear(self.joint_num, 1, bias=False).cuda()

		self.spatial_JCW4 = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.temporal_JCW4 = nn.Linear(self.joint_num, 1, bias=False).cuda()
		# Feature Encoders
		self.encode_diffwhole_512_rgb = nn.Sequential(
			nn.Linear(int(self.whole_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_diffwhole_512_flow = nn.Sequential(
			nn.Linear(int(self.whole_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_whole_512_rgb = nn.Sequential(
			nn.Linear(int(self.whole_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_whole_512_flow = nn.Sequential(
			nn.Linear(int(self.whole_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Comm0_512_rgb = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Comm0_512_flow = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()

		self.encode_Comm1_512_Con1_rgb = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Comm1_512_Con1_flow = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Comm1_512_Con2_rgb = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Comm1_512_Con2_flow = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Comm1_512_Con3_rgb = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Comm1_512_Con3_flow = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Comm1_512_Con4_rgb = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Comm1_512_Con4_flow = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()

		self.encode_Diff0_512_Con1_rgb = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Diff0_512_Con1_flow = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Diff0_512_Con2_rgb = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Diff0_512_Con2_flow = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Diff0_512_Con3_rgb = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Diff0_512_Con3_flow = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Diff0_512_Con4_rgb = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Diff0_512_Con4_flow = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()

		self.encode_Diff1_512_Con1_rgb = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Diff1_512_Con1_flow = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Diff1_512_Con2_rgb = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True)
			).cuda()
		self.encode_Diff1_512_Con2_flow = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True)
			).cuda()
		self.encode_Diff1_512_Con3_rgb = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Diff1_512_Con3_flow = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Diff1_512_Con4_rgb = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()
		self.encode_Diff1_512_Con4_flow = nn.Sequential(
			nn.Linear(int(self.patch_size/2), int(self.hidden1/2)),
			nn.Dropout(self.dropout_rate),
			nn.ReLU(inplace=True),
			).cuda()

		# NAS selectors
		self.operation_selector01 = nn.Linear(7,1, bias=False).cuda()
		self.operation_selector02 = nn.Linear(7,1, bias=False).cuda()
		self.operation_selector03 = nn.Linear(7,1, bias=False).cuda()
		self.operation_selector12 = nn.Linear(7,1, bias=False).cuda()
		self.operation_selector13 = nn.Linear(7,1, bias=False).cuda()
		self.operation_selector23 = nn.Linear(7,1, bias=False).cuda()

		# NAS Operations
		self.FC_Reduce01_joint_rgb = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.FC_Reduce02_joint_rgb = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.FC_Reduce03_joint_rgb = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.FC_Reduce12_joint_rgb = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.FC_Reduce13_joint_rgb = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.FC_Reduce23_joint_rgb = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.FC_Reduce01_joint_flow = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.FC_Reduce02_joint_flow = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.FC_Reduce03_joint_flow = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.FC_Reduce12_joint_flow = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.FC_Reduce13_joint_flow = nn.Linear(self.joint_num, 1, bias=False).cuda()
		self.FC_Reduce23_joint_flow = nn.Linear(self.joint_num, 1, bias=False).cuda()

		self.FC_Reduce01_module_rgb = nn.Linear(self.module_num, 1, bias=False).cuda()
		self.FC_Reduce02_module_rgb = nn.Linear(self.module_num, 1, bias=False).cuda()
		self.FC_Reduce03_module_rgb = nn.Linear(self.module_num, 1, bias=False).cuda()
		self.FC_Reduce12_module_rgb = nn.Linear(self.module_num, 1, bias=False).cuda()
		self.FC_Reduce13_module_rgb = nn.Linear(self.module_num, 1, bias=False).cuda()
		self.FC_Reduce23_module_rgb = nn.Linear(self.module_num, 1, bias=False).cuda()
		self.FC_Reduce01_module_flow = nn.Linear(self.module_num, 1, bias=False).cuda()
		self.FC_Reduce02_module_flow = nn.Linear(self.module_num, 1, bias=False).cuda()
		self.FC_Reduce03_module_flow = nn.Linear(self.module_num, 1, bias=False).cuda()
		self.FC_Reduce12_module_flow = nn.Linear(self.module_num, 1, bias=False).cuda()
		self.FC_Reduce13_module_flow = nn.Linear(self.module_num, 1, bias=False).cuda()
		self.FC_Reduce23_module_flow = nn.Linear(self.module_num, 1, bias=False).cuda()

		self.conv01_max_joint_rgb = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv02_max_joint_rgb = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv03_max_joint_rgb = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv12_max_joint_rgb = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv13_max_joint_rgb = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv23_max_joint_rgb = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv01_max_joint_flow = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv02_max_joint_flow = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv03_max_joint_flow = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv12_max_joint_flow = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv13_max_joint_flow = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv23_max_joint_flow = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()

		self.conv01_max_module_rgb = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv02_max_module_rgb = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv03_max_module_rgb = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv12_max_module_rgb = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv13_max_module_rgb = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv23_max_module_rgb = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv01_max_module_flow = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv02_max_module_flow = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv03_max_module_flow = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv12_max_module_flow = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv13_max_module_flow = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv23_max_module_flow = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		
		self.conv01_Red_joint_rgb = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv02_Red_joint_rgb = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv03_Red_joint_rgb = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv12_Red_joint_rgb = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv13_Red_joint_rgb = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv23_Red_joint_rgb = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv01_Red_joint_flow = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv02_Red_joint_flow = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv03_Red_joint_flow = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv12_Red_joint_flow = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv13_Red_joint_flow = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv23_Red_joint_flow = nn.Conv2d(in_channels=1, out_channels=self.joint_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()

		self.conv01_Red_module_rgb = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv02_Red_module_rgb = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv03_Red_module_rgb = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv12_Red_module_rgb = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv13_Red_module_rgb = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv23_Red_module_rgb = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv01_Red_module_flow = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv02_Red_module_flow = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv03_Red_module_flow = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv12_Red_module_flow = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv13_Red_module_flow = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv23_Red_module_flow = nn.Conv2d(in_channels=1, out_channels=self.module_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()

		self.conv01_max_time_rgb = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv02_max_time_rgb = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv03_max_time_rgb = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv12_max_time_rgb = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv13_max_time_rgb = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv23_max_time_rgb = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv01_max_time_flow = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv02_max_time_flow = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv03_max_time_flow = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv12_max_time_flow = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv13_max_time_flow = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv23_max_time_flow = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()

		self.FC_Reduce01_time_rgb = nn.Linear(self.seg_num, 1, bias=False).cuda()
		self.FC_Reduce02_time_rgb = nn.Linear(self.seg_num, 1, bias=False).cuda()
		self.FC_Reduce03_time_rgb = nn.Linear(self.seg_num, 1, bias=False).cuda()
		self.FC_Reduce12_time_rgb = nn.Linear(self.seg_num, 1, bias=False).cuda()
		self.FC_Reduce13_time_rgb = nn.Linear(self.seg_num, 1, bias=False).cuda()
		self.FC_Reduce23_time_rgb = nn.Linear(self.seg_num, 1, bias=False).cuda()
		self.FC_Reduce01_time_flow = nn.Linear(self.seg_num, 1, bias=False).cuda()
		self.FC_Reduce02_time_flow = nn.Linear(self.seg_num, 1, bias=False).cuda()
		self.FC_Reduce03_time_flow = nn.Linear(self.seg_num, 1, bias=False).cuda()
		self.FC_Reduce12_time_flow = nn.Linear(self.seg_num, 1, bias=False).cuda()
		self.FC_Reduce13_time_flow = nn.Linear(self.seg_num, 1, bias=False).cuda()
		self.FC_Reduce23_time_flow = nn.Linear(self.seg_num, 1, bias=False).cuda()

		self.conv01_Red_time_rgb = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv02_Red_time_rgb = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv03_Red_time_rgb = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv12_Red_time_rgb = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv13_Red_time_rgb = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv23_Red_time_rgb = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv01_Red_time_flow = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv02_Red_time_flow = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv03_Red_time_flow = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv12_Red_time_flow = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv13_Red_time_flow = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()
		self.conv23_Red_time_flow = nn.Conv2d(in_channels=1, out_channels=self.seg_num, kernel_size=1, stride=1, padding=0, bias=False).cuda()

		self.attention_01_rgb = nn.Sequential(
			nn.Linear(int(self.hidden1/2), int(self.hidden1/4.0)),
			nn.Tanh(),
			nn.Linear(int(self.hidden1/4.0), 1),
			nn.Softmax(dim=1)).cuda()
		self.attention_02_rgb = nn.Sequential(
			nn.Linear(int(self.hidden1/2), int(self.hidden1/4.0)),
			nn.Tanh(),
			nn.Linear(int(self.hidden1/4.0), 1),
			nn.Softmax(dim=1)).cuda()
		self.attention_03_rgb = nn.Sequential(
			nn.Linear(int(self.hidden1/2), int(self.hidden1/4.0)),
			nn.Tanh(),
			nn.Linear(int(self.hidden1/4.0), 1),
			nn.Softmax(dim=1)).cuda()
		self.attention_12_rgb = nn.Sequential(
			nn.Linear(int(self.hidden1/2), int(self.hidden1/4.0)),
			nn.Tanh(),
			nn.Linear(int(self.hidden1/4.0), 1),
			nn.Softmax(dim=1)).cuda()
		self.attention_13_rgb = nn.Sequential(
			nn.Linear(int(self.hidden1/2), int(self.hidden1/4.0)),
			nn.Tanh(),
			nn.Linear(int(self.hidden1/4.0), 1),
			nn.Softmax(dim=1)).cuda()
		self.attention_23_rgb = nn.Sequential(
			nn.Linear(int(self.hidden1/2), int(self.hidden1/4.0)),
			nn.Tanh(),
			nn.Linear(int(self.hidden1/4.0), 1),
			nn.Softmax(dim=1)).cuda()
		self.attention_01_flow = nn.Sequential(
			nn.Linear(int(self.hidden1/2), int(self.hidden1/4.0)),
			nn.Tanh(),
			nn.Linear(int(self.hidden1/4.0), 1),
			nn.Softmax(dim=1)).cuda()
		self.attention_02_flow = nn.Sequential(
			nn.Linear(int(self.hidden1/2), int(self.hidden1/4.0)),
			nn.Tanh(),
			nn.Linear(int(self.hidden1/4.0), 1),
			nn.Softmax(dim=1)).cuda()
		self.attention_03_flow = nn.Sequential(
			nn.Linear(int(self.hidden1/2), int(self.hidden1/4.0)),
			nn.Tanh(),
			nn.Linear(int(self.hidden1/4.0), 1),
			nn.Softmax(dim=1)).cuda()
		self.attention_12_flow = nn.Sequential(
			nn.Linear(int(self.hidden1/2), int(self.hidden1/4.0)),
			nn.Tanh(),
			nn.Linear(int(self.hidden1/4.0), 1),
			nn.Softmax(dim=1)).cuda()
		self.attention_13_flow = nn.Sequential(
			nn.Linear(int(self.hidden1/2), int(self.hidden1/4.0)),
			nn.Tanh(),
			nn.Linear(int(self.hidden1/4.0), 1),
			nn.Softmax(dim=1)).cuda()
		self.attention_23_flow = nn.Sequential(
			nn.Linear(int(self.hidden1/2), int(self.hidden1/4.0)),
			nn.Tanh(),
			nn.Linear(int(self.hidden1/4.0), 1),
			nn.Softmax(dim=1)).cuda()
		
		self.conv3d01_rgb = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False).cuda()
		self.conv3d02_rgb = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False).cuda()
		self.conv3d03_rgb = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False).cuda()
		self.conv3d12_rgb = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False).cuda()
		self.conv3d13_rgb = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False).cuda()
		self.conv3d23_rgb = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False).cuda()
		self.conv3d01_flow = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False).cuda()
		self.conv3d02_flow = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False).cuda()
		self.conv3d03_flow = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False).cuda()
		self.conv3d12_flow = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False).cuda()
		self.conv3d13_flow = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False).cuda()
		self.conv3d23_flow = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False).cuda()

		# Assessment Module
		self.assessment1 = nn.Sequential(
			nn.ReLU(inplace=True),
			#nn.BatchNorm1d(self.hidden1*2),
			nn.Linear(self.hidden1, self.hidden2),
			nn.Dropout(self.dropout_rate)).cuda()
		self.assessment2 = nn.Sequential(
			nn.ReLU(inplace=True),
			#nn.BatchNorm1d(self.hidden2),
			nn.Linear(self.hidden2, 1),
			nn.Dropout(self.dropout_rate)).cuda()

		# Classification Module
		self.classify1 = nn.Sequential(
			nn.ReLU(inplace=True),
			#nn.BatchNorm1d(512),
			nn.Linear(self.hidden1, self.hidden2),
			nn.Dropout(self.dropout_rate)).cuda()
		self.classify2 = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Linear(self.hidden2, 1),
			nn.Dropout(self.dropout_rate)).cuda()

		self.last_fuse = nn.Linear(self.module_num, 1, bias=False).cuda() #nn.AdaptiveAvgPool2d((2,2)).cuda()
			
		
	def Out_Nas(self, Cube_in_rgb, Cube_in_flow, Conv_max_joint_rgb, Conv_max_module_rgb, \
			FC_Reduce_joint_rgb, FC_Reduce_module_rgb,  Conv_Red_joint_rgb, Conv_Red_module_rgb, \
			Conv_max_time_rgb, FC_Reduce_time_rgb, Conv_Red_time_rgb, attention_rgb, Conv_3D_rgb, \
			Conv_max_joint_flow, Conv_max_module_flow, \
			FC_Reduce_joint_flow, FC_Reduce_module_flow,  Conv_Red_joint_flow, Conv_Red_module_flow, \
			Conv_max_time_flow, FC_Reduce_time_flow, Conv_Red_time_flow, attention_flow, Conv_3D_flow, op_select):
		import pdb

		Cube_out_max2d_rgb = \
			Conv_max_module_rgb(torch.max(\
			Conv_max_joint_rgb(torch.max(Cube_in_rgb.permute(0,4,2,1,3).reshape(-1, self.joint_num, self.seg_num, self.module_num), dim=1, keepdim=True)[0]).permute(0,3,1,2) # (-1*hidden1, module, joint, seg)
			, dim=1, keepdim=True)[0]).permute(0,3,1,2).view(-1,  int(self.hidden1/2), self.seg_num, self.module_num, self.joint_num).permute(0,2,4,3,1)
		
		Cube_out_max2d_flow = \
			Conv_max_module_flow(torch.max(\
			Conv_max_joint_flow(torch.max(Cube_in_flow.permute(0,4,2,1,3).reshape(-1, self.joint_num, self.seg_num, self.module_num), dim=1, keepdim=True)[0]).permute(0,3,1,2) # (-1*hidden1, module, joint, seg)
			, dim=1, keepdim=True)[0]).permute(0,3,1,2).view(-1,  int(self.hidden1/2), self.seg_num, self.module_num, self.joint_num).permute(0,2,4,3,1)

		
		Cube_out_Red_rgb = Conv_Red_time_rgb(torch.matmul(\
			Conv_Red_module_rgb(torch.matmul(\
			Conv_Red_joint_rgb(torch.matmul(Cube_in_rgb.permute(0,4,1,3,2).reshape(-1, self.seg_num, self.module_num, self.joint_num), FC_Reduce_joint_rgb.weight.view(self.joint_num,1)).permute(0,3,1,2))\
			, FC_Reduce_module_rgb.weight.view(self.module_num,1)).permute(0,3,1,2))\
			, FC_Reduce_time_rgb.weight.view(self.seg_num,1)).permute(0,3,2,1)).reshape(-1, int(self.hidden1/2), self.seg_num, self.joint_num, self.module_num).permute(0,2,3,4,1)
		
		Cube_out_Red_flow = Conv_Red_time_flow(torch.matmul(\
			Conv_Red_module_flow(torch.matmul(\
			Conv_Red_joint_flow(torch.matmul(Cube_in_flow.permute(0,4,1,3,2).reshape(-1, self.seg_num, self.module_num, self.joint_num), FC_Reduce_joint_flow.weight.view(self.joint_num,1)).permute(0,3,1,2))\
			, FC_Reduce_module_flow.weight.view(self.module_num,1)).permute(0,3,1,2))\
			, FC_Reduce_time_flow.weight.view(self.seg_num,1)).permute(0,3,2,1)).reshape(-1, int(self.hidden1/2), self.seg_num, self.joint_num, self.module_num).permute(0,2,3,4,1)
		
		Cube_out_conv3d_rgb = Conv_3D_rgb(Cube_in_rgb.permute(0,4,1,2,3).reshape(-1, 1, self.seg_num, self.joint_num, self.module_num)).reshape(-1,  int(self.hidden1/2), self.seg_num, self.joint_num, self.module_num).permute(0,2,3,4,1)

		Cube_out_conv3d_flow = Conv_3D_flow(Cube_in_flow.permute(0,4,1,2,3).reshape(-1, 1, self.seg_num, self.joint_num, self.module_num)).reshape(-1,  int(self.hidden1/2), self.seg_num, self.joint_num, self.module_num).permute(0,2,3,4,1)

			
		Cube_out_max_time_rgb = Conv_max_time_rgb(torch.max(\
			Cube_in_rgb.permute(0,4,1,2,3).reshape(-1, self.seg_num, self.joint_num, self.module_num)\
			, dim=1, keepdim=True)[0]).view(-1,  int(self.hidden1/2), self.seg_num, self.joint_num, self.module_num).permute(0,2,3,4,1)
		
		Cube_out_max_time_flow = Conv_max_time_flow(torch.max(\
			Cube_in_flow.permute(0,4,1,2,3).reshape(-1, self.seg_num, self.joint_num, self.module_num)\
			, dim=1, keepdim=True)[0]).view(-1,  int(self.hidden1/2), self.seg_num, self.joint_num, self.module_num).permute(0,2,3,4,1)
		
		Cube_out_att_rgb = Cube_in_rgb * attention_rgb(Cube_in_rgb.permute(0,2,3,1,4).reshape(-1, self.joint_num*self.module_num*self.seg_num,  int(self.hidden1/2))).reshape(-1, self.joint_num, self.module_num, self.seg_num, 1).permute(0,3,1,2,4)

		Cube_out_att_flow = Cube_in_flow * attention_flow(Cube_in_flow.permute(0,2,3,1,4).reshape(-1, self.joint_num*self.module_num*self.seg_num,  int(self.hidden1/2))).reshape(-1, self.joint_num, self.module_num, self.seg_num, 1).permute(0,3,1,2,4)

		Cube_out_iden_rgb = Cube_in_rgb
		Cube_out_iden_flow = Cube_in_flow
		
		shape_ = Cube_in_rgb.shape
		Cube_out_Zero_rgb = torch.cuda.FloatTensor(shape_) if torch.cuda.is_available() else torch.FloatTensor(shape_)
		torch.zeros(shape_, out=Cube_out_Zero_rgb)
		Cube_out_Zero_flow = torch.cuda.FloatTensor(shape_) if torch.cuda.is_available() else torch.FloatTensor(shape_)
		torch.zeros(shape_, out=Cube_out_Zero_flow)
		
		if self.mode in ['structure', 'random']:
			selector = op_select.weight.view(7, 1)
		elif self.mode == 'choice':
			selector = F.softmax(op_select.weight.view(7, 1), dim=0)
		Cube_out_rgb = torch.matmul(torch.cat((
			Cube_out_max2d_rgb, Cube_out_Red_rgb, \
			Cube_out_conv3d_rgb, \
			Cube_out_max_time_rgb, Cube_out_att_rgb, Cube_out_iden_rgb, Cube_out_Zero_rgb), dim=4).reshape(-1, self.seg_num, self.joint_num, self.module_num, 7,  int(self.hidden1/2)).permute(0,1,2,3,5,4), \
			selector).reshape(-1, self.seg_num, self.joint_num, self.module_num,  int(self.hidden1/2))
		Cube_out_flow = torch.matmul(torch.cat((
			Cube_out_max2d_flow, Cube_out_Red_flow, \
			Cube_out_conv3d_flow, \
			Cube_out_max_time_flow, Cube_out_att_flow, Cube_out_iden_flow, Cube_out_Zero_flow), dim=4).reshape(-1, self.seg_num, self.joint_num, self.module_num, 7,  int(self.hidden1/2)).permute(0,1,2,3,5,4), \
			selector).reshape(-1, self.seg_num, self.joint_num, self.module_num,  int(self.hidden1/2))
		return Cube_out_rgb, Cube_out_flow

	def Final_Nas(self, Cube_in_rgb, Cube_in_flow, Conv_max_joint_rgb, Conv_max_module_rgb, \
			FC_Reduce_joint_rgb, FC_Reduce_module_rgb,  Conv_Red_joint_rgb, Conv_Red_module_rgb, \
			Conv_max_time_rgb, FC_Reduce_time_rgb, Conv_Red_time_rgb, attention_rgb, Conv_3D_rgb, \
			Conv_max_joint_flow, Conv_max_module_flow, \
			FC_Reduce_joint_flow, FC_Reduce_module_flow,  Conv_Red_joint_flow, Conv_Red_module_flow, \
			Conv_max_time_flow, FC_Reduce_time_flow, Conv_Red_time_flow, attention_flow, Conv_3D_flow, op_select):
		shape_ = Cube_in_rgb.shape
		Cube_out_Zero_rgb = torch.cuda.FloatTensor(shape_) if torch.cuda.is_available() else torch.FloatTensor(shape_)
		torch.zeros(shape_, out=Cube_out_Zero_rgb)
		Cube_out_Zero_flow = torch.cuda.FloatTensor(shape_) if torch.cuda.is_available() else torch.FloatTensor(shape_)
		torch.zeros(shape_, out=Cube_out_Zero_flow)
		Cube_out_rgb = Cube_out_Zero_rgb
		Cube_out_flow = Cube_out_Zero_flow
		if op_select[0] == 1:
			Cube_out_max2d_rgb = \
				Conv_max_module_rgb(torch.max(\
				Conv_max_joint_rgb(torch.max(Cube_in_rgb.permute(0,4,2,1,3).reshape(-1, self.joint_num, self.seg_num, self.module_num), dim=1, keepdim=True)[0]).permute(0,3,1,2) # (-1*hidden1, module, joint, seg)
				, dim=1, keepdim=True)[0]).permute(0,3,1,2).view(-1,  int(self.hidden1/2), self.seg_num, self.module_num, self.joint_num).permute(0,2,4,3,1)
			Cube_out_max2d_flow = \
				Conv_max_module_flow(torch.max(\
				Conv_max_joint_flow(torch.max(Cube_in_flow.permute(0,4,2,1,3).reshape(-1, self.joint_num, self.seg_num, self.module_num), dim=1, keepdim=True)[0]).permute(0,3,1,2) # (-1*hidden1, module, joint, seg)
				, dim=1, keepdim=True)[0]).permute(0,3,1,2).view(-1,  int(self.hidden1/2), self.seg_num, self.module_num, self.joint_num).permute(0,2,4,3,1)
			Cube_out_rgb += Cube_out_max2d_rgb
			Cube_out_flow += Cube_out_max2d_flow
		if op_select[1] == 1:
			Cube_out_Red_rgb = Conv_Red_time_rgb(torch.matmul(\
				Conv_Red_module_rgb(torch.matmul(\
				Conv_Red_joint_rgb(torch.matmul(Cube_in_rgb.permute(0,4,1,3,2).reshape(-1, self.seg_num, self.module_num, self.joint_num), FC_Reduce_joint_rgb.weight.view(self.joint_num,1)).permute(0,3,1,2))\
				, FC_Reduce_module_rgb.weight.view(self.module_num,1)).permute(0,3,1,2))\
				, FC_Reduce_time_rgb.weight.view(self.seg_num,1)).permute(0,3,2,1)).reshape(-1, int(self.hidden1/2), self.seg_num, self.joint_num, self.module_num).permute(0,2,3,4,1)
			Cube_out_Red_flow = Conv_Red_time_flow(torch.matmul(\
				Conv_Red_module_flow(torch.matmul(\
				Conv_Red_joint_flow(torch.matmul(Cube_in_flow.permute(0,4,1,3,2).reshape(-1, self.seg_num, self.module_num, self.joint_num), FC_Reduce_joint_flow.weight.view(self.joint_num,1)).permute(0,3,1,2))\
				, FC_Reduce_module_flow.weight.view(self.module_num,1)).permute(0,3,1,2))\
				, FC_Reduce_time_flow.weight.view(self.seg_num,1)).permute(0,3,2,1)).reshape(-1, int(self.hidden1/2), self.seg_num, self.joint_num, self.module_num).permute(0,2,3,4,1)
			Cube_out_rgb += Cube_out_Red_rgb
			Cube_out_flow += Cube_out_Red_flow
		if op_select[2] == 1:
			Cube_out_conv3d_rgb = Conv_3D_rgb(Cube_in_rgb.permute(0,4,1,2,3).reshape(-1, 1, self.seg_num, self.joint_num, self.module_num)).reshape(-1,  int(self.hidden1/2), self.seg_num, self.joint_num, self.module_num).permute(0,2,3,4,1)
			Cube_out_conv3d_flow = Conv_3D_flow(Cube_in_flow.permute(0,4,1,2,3).reshape(-1, 1, self.seg_num, self.joint_num, self.module_num)).reshape(-1,  int(self.hidden1/2), self.seg_num, self.joint_num, self.module_num).permute(0,2,3,4,1)
			Cube_out_rgb += Cube_out_conv3d_rgb
			Cube_out_flow += Cube_out_conv3d_flow
		if op_select[3] == 1:
			Cube_out_max_time_rgb = Conv_max_time_rgb(torch.max(\
				Cube_in_rgb.permute(0,4,1,2,3).reshape(-1, self.seg_num, self.joint_num, self.module_num)\
				, dim=1, keepdim=True)[0]).view(-1,  int(self.hidden1/2), self.seg_num, self.joint_num, self.module_num).permute(0,2,3,4,1)
			Cube_out_max_time_flow = Conv_max_time_flow(torch.max(\
				Cube_in_flow.permute(0,4,1,2,3).reshape(-1, self.seg_num, self.joint_num, self.module_num)\
				, dim=1, keepdim=True)[0]).view(-1,  int(self.hidden1/2), self.seg_num, self.joint_num, self.module_num).permute(0,2,3,4,1)
			Cube_out_rgb += Cube_out_max_time_rgb
			Cube_out_flow += Cube_out_max_time_flow
		if op_select[4] == 1:
			Cube_out_att_rgb = Cube_in_rgb * attention_rgb(Cube_in_rgb.permute(0,2,3,1,4).reshape(-1, self.joint_num*self.module_num*self.seg_num,  int(self.hidden1/2))).reshape(-1, self.joint_num, self.module_num, self.seg_num, 1).permute(0,3,1,2,4)
			Cube_out_att_flow = Cube_in_flow * attention_flow(Cube_in_flow.permute(0,2,3,1,4).reshape(-1, self.joint_num*self.module_num*self.seg_num,  int(self.hidden1/2))).reshape(-1, self.joint_num, self.module_num, self.seg_num, 1).permute(0,3,1,2,4)
			Cube_out_rgb += Cube_out_att_rgb
			Cube_out_flow += Cube_out_att_flow
		if op_select[5] == 1:
			Cube_out_iden_rgb = Cube_in_rgb
			Cube_out_iden_flow = Cube_in_flow
			Cube_out_rgb += Cube_out_iden_rgb
			Cube_out_flow += Cube_out_iden_flow 
		return Cube_out_rgb, Cube_out_flow

	def get_Nas(self, Cube_in_rgb, Cube_in_flow, Conv_max_joint_rgb, Conv_max_module_rgb, \
			FC_Reduce_joint_rgb, FC_Reduce_module_rgb,  Conv_Red_joint_rgb, Conv_Red_module_rgb, \
			Conv_max_time_rgb, FC_Reduce_time_rgb, Conv_Red_time_rgb, attention_rgb, Conv_3D_rgb, \
			Conv_max_joint_flow, Conv_max_module_flow, \
			FC_Reduce_joint_flow, FC_Reduce_module_flow,  Conv_Red_joint_flow, Conv_Red_module_flow, \
			Conv_max_time_flow, FC_Reduce_time_flow, Conv_Red_time_flow, attention_flow, Conv_3D_flow, op_select):
		if self.mode in ['structure', 'random']:
			return self.Final_Nas(Cube_in_rgb, Cube_in_flow, Conv_max_joint_rgb, Conv_max_module_rgb, \
				FC_Reduce_joint_rgb, FC_Reduce_module_rgb,  Conv_Red_joint_rgb, Conv_Red_module_rgb, \
				Conv_max_time_rgb, FC_Reduce_time_rgb, Conv_Red_time_rgb, attention_rgb, Conv_3D_rgb, \
				Conv_max_joint_flow, Conv_max_module_flow, \
				FC_Reduce_joint_flow, FC_Reduce_module_flow,  Conv_Red_joint_flow, Conv_Red_module_flow, \
				Conv_max_time_flow, FC_Reduce_time_flow, Conv_Red_time_flow, attention_flow, Conv_3D_flow, op_select.weight[0])
		elif self.mode == 'choice':
			return self.Out_Nas(Cube_in_rgb, Cube_in_flow, Conv_max_joint_rgb, Conv_max_module_rgb, \
				FC_Reduce_joint_rgb, FC_Reduce_module_rgb,  Conv_Red_joint_rgb, Conv_Red_module_rgb, \
				Conv_max_time_rgb, FC_Reduce_time_rgb, Conv_Red_time_rgb, attention_rgb, Conv_3D_rgb, \
				Conv_max_joint_flow, Conv_max_module_flow, \
				FC_Reduce_joint_flow, FC_Reduce_module_flow,  Conv_Red_joint_flow, Conv_Red_module_flow, \
				Conv_max_time_flow, FC_Reduce_time_flow, Conv_Red_time_flow, attention_flow, Conv_3D_flow, op_select)

	def forward(self, loss_type, feat_whole, feat_patch, truth):
		# Define graph conv
		self.connectivity_graph1 = torch.tensor(self.a).float().cuda()
		self.connectivity_graph2 = torch.tensor(self.a2).float().cuda()
		self.connectivity_graph3 = torch.tensor(self.a3).float().cuda()
		self.connectivity_graph4 = torch.tensor(self.a4).float().cuda()

		self.spatial_graph1 = torch.abs(self.spatial_mat1.weight*self.connectivity_graph1).cuda()
		self.spatial_graph2 = torch.abs(self.spatial_mat2.weight*self.connectivity_graph2).cuda()
		self.spatial_graph3 = torch.abs(self.spatial_mat3.weight*self.connectivity_graph3).cuda()
		self.spatial_graph4 = torch.abs(self.spatial_mat4.weight*self.connectivity_graph4).cuda()

		self.temporal_graph1 = torch.abs(self.temporal_mat1.weight*self.connectivity_graph1).cuda()
		self.temporal_graph2 = torch.abs(self.temporal_mat2.weight*self.connectivity_graph2).cuda()
		self.temporal_graph3 = torch.abs(self.temporal_mat3.weight*self.connectivity_graph3).cuda()
		self.temporal_graph4 = torch.abs(self.temporal_mat4.weight*self.connectivity_graph4).cuda()

		# Commonality Module
		Commonality_H0 = feat_patch.permute(0,2,3,1)

		Commonality_H1_Con1 = torch.matmul(feat_patch.permute(0,2,3,1).reshape(-1, self.joint_num), 
			self.spatial_graph1).reshape(-1, self.seg_num, self.patch_size, self.joint_num)
		Commonality_H1_Con2 = torch.matmul(feat_patch.permute(0,2,3,1).reshape(-1, self.joint_num), 
			self.spatial_graph2).reshape(-1, self.seg_num, self.patch_size, self.joint_num)
		Commonality_H1_Con3 = torch.matmul(feat_patch.permute(0,2,3,1).reshape(-1, self.joint_num), 
			self.spatial_graph3).reshape(-1, self.seg_num, self.patch_size, self.joint_num)
		Commonality_H1_Con4 = torch.matmul(feat_patch.permute(0,2,3,1).reshape(-1, self.joint_num), 
			self.spatial_graph4).reshape(-1, self.seg_num, self.patch_size, self.joint_num)
		
		Comm_h0 = Commonality_H0.permute((0,1,3,2))
		Comm_h1_Con1 = Commonality_H1_Con1.permute((0,1,3,2))
		Comm_h1_Con2 = Commonality_H1_Con2.permute((0,1,3,2))
		Comm_h1_Con3 = Commonality_H1_Con3.permute((0,1,3,2))
		Comm_h1_Con4 = Commonality_H1_Con4.permute((0,1,3,2))
		# Difference Module
		diff_mat_Fp0 = feat_patch.permute(0,2,3,1) # shape(-1, self.seg_num, self.patch_size, self.joint_num)
		diff_mat_Fp1 = torch.cat((feat_patch.permute(0,2,3,1)[:,1:self.seg_num,:,:], 
			feat_patch.permute(0,2,3,1)[:,self.seg_num-1,:,:].reshape(-1, 1, self.patch_size, self.joint_num)), dim=1) # shape(-1, self.seg_num, self.patch_size, self.joint_num)
		diff_mat_F0 = diff_mat_Fp0[..., None, :] - diff_mat_Fp0[..., None] # shape(-1, self.seg_num, self.patch_size, self.joint_num, self.joint_num)
		diff_mat_F1 = diff_mat_Fp1[..., None, :] - diff_mat_Fp0[..., None] # shape(-1, self.seg_num, self.patch_size, self.joint_num, self.joint_num)

		Difference_D0_Con1 = self.spatial_JCW1(diff_mat_F0 * self.spatial_graph1).reshape(-1, self.seg_num, self.patch_size, self.joint_num) # shape(-1, self.seg_num, self.patch_size, self.joint_num)
		Difference_D0_Con2 = self.spatial_JCW2(diff_mat_F0 * self.spatial_graph2).reshape(-1, self.seg_num, self.patch_size, self.joint_num)  # shape(-1, self.seg_num, self.patch_size, self.joint_num)
		Difference_D0_Con3 = self.spatial_JCW3(diff_mat_F0 * self.spatial_graph3).reshape(-1, self.seg_num, self.patch_size, self.joint_num)  # shape(-1, self.seg_num, self.patch_size, self.joint_num)
		Difference_D0_Con4 = self.spatial_JCW3(diff_mat_F0 * self.spatial_graph4).reshape(-1, self.seg_num, self.patch_size, self.joint_num)  # shape(-1, self.seg_num, self.patch_size, self.joint_num)


		Difference_D1_Con1 = self.temporal_JCW1(diff_mat_F1 * self.temporal_graph1).reshape(-1, self.seg_num, self.patch_size, self.joint_num)  # shape(-1, self.seg_num, self.patch_size, self.joint_num)
		Difference_D1_Con2 = self.temporal_JCW2(diff_mat_F1 * self.temporal_graph2).reshape(-1, self.seg_num, self.patch_size, self.joint_num)  # shape(-1, self.seg_num, self.patch_size, self.joint_num)
		Difference_D1_Con3 = self.temporal_JCW3(diff_mat_F1 * self.temporal_graph3).reshape(-1, self.seg_num, self.patch_size, self.joint_num)  # shape(-1, self.seg_num, self.patch_size, self.joint_num)
		Difference_D1_Con4 = self.temporal_JCW3(diff_mat_F1 * self.temporal_graph4).reshape(-1, self.seg_num, self.patch_size, self.joint_num)  # shape(-1, self.seg_num, self.patch_size, self.joint_num)
	
		Diff_d0_Con1 = Difference_D0_Con1.permute((0,1,3,2))
		Diff_d0_Con2 = Difference_D0_Con2.permute((0,1,3,2))
		Diff_d0_Con3 = Difference_D0_Con3.permute((0,1,3,2))
		Diff_d0_Con4 = Difference_D0_Con4.permute((0,1,3,2))
		Diff_d1_Con1 = Difference_D1_Con1.permute((0,1,3,2))
		Diff_d1_Con2 = Difference_D1_Con2.permute((0,1,3,2))
		Diff_d1_Con3 = Difference_D1_Con3.permute((0,1,3,2))
		Diff_d1_Con4 = Difference_D1_Con4.permute((0,1,3,2))

		# Diff Whole
		diff_mat_Fp0 = feat_whole
		diff_mat_Fp1 = torch.cat((feat_whole[:,1:feat_whole.shape[1],:], 
			feat_whole[:,feat_whole.shape[1]-1,:].reshape(-1, 1, self.whole_size)), dim=1)
		feat_diff = torch.abs(diff_mat_Fp1 - diff_mat_Fp0)
		
		# Encoding shape(batch, seg_num, joint_num, hidden1)
		encoded_whole = torch.cat((\
			self.encode_whole_512_rgb(feat_whole[:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, 1, int(self.hidden1/2)).repeat(1, 1, self.joint_num, 1).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_whole_512_flow(feat_whole[:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, 1, int(self.hidden1/2)).repeat(1, 1, self.joint_num, 1).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_diff = torch.cat((\
			self.encode_diffwhole_512_rgb(feat_diff[:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, 1, int(self.hidden1/2)).repeat(1, 1, self.joint_num, 1).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_diffwhole_512_flow(feat_diff[:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, 1, int(self.hidden1/2)).repeat(1, 1, self.joint_num, 1).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_Comm0 = torch.cat((\
			self.encode_Comm0_512_rgb(Comm_h0[:,:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_Comm0_512_flow(Comm_h0[:,:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_Comm1_Con1 = torch.cat((\
			self.encode_Comm0_512_rgb(Comm_h1_Con1[:,:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_Comm0_512_flow(Comm_h1_Con1[:,:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_Comm1_Con2 = torch.cat((\
			self.encode_Comm0_512_rgb(Comm_h1_Con2[:,:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_Comm0_512_flow(Comm_h1_Con2[:,:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_Comm1_Con3 = torch.cat((\
			self.encode_Comm0_512_rgb(Comm_h1_Con3[:,:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_Comm0_512_flow(Comm_h1_Con3[:,:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_Comm1_Con4 = torch.cat((\
			self.encode_Comm0_512_rgb(Comm_h1_Con4[:,:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_Comm0_512_flow(Comm_h1_Con4[:,:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_Diff0_Con1 = torch.cat((\
			self.encode_Diff0_512_Con1_rgb(Diff_d0_Con1[:,:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_Diff0_512_Con1_flow(Diff_d0_Con1[:,:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_Diff0_Con2 = torch.cat((\
			self.encode_Diff0_512_Con1_rgb(Diff_d0_Con2[:,:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_Diff0_512_Con1_flow(Diff_d0_Con2[:,:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_Diff0_Con3 = torch.cat((\
			self.encode_Diff0_512_Con1_rgb(Diff_d0_Con3[:,:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_Diff0_512_Con1_flow(Diff_d0_Con3[:,:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_Diff0_Con4 = torch.cat((\
			self.encode_Diff0_512_Con1_rgb(Diff_d0_Con4[:,:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_Diff0_512_Con1_flow(Diff_d0_Con4[:,:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_Diff1_Con1 = torch.cat((\
			self.encode_Diff0_512_Con1_rgb(Diff_d1_Con1[:,:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_Diff0_512_Con1_flow(Diff_d1_Con1[:,:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_Diff1_Con2 = torch.cat((\
			self.encode_Diff0_512_Con1_rgb(Diff_d1_Con2[:,:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_Diff0_512_Con1_flow(Diff_d1_Con2[:,:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_Diff1_Con3 = torch.cat((\
			self.encode_Diff0_512_Con1_rgb(Diff_d1_Con3[:,:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_Diff0_512_Con1_flow(Diff_d1_Con3[:,:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)
		encoded_Diff1_Con4 = torch.cat((\
			self.encode_Diff0_512_Con1_rgb(Diff_d1_Con4[:,:,:,0:int(self.whole_size/2)]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2)),
			self.encode_Diff0_512_Con1_flow(Diff_d1_Con4[:,:,:,int(self.whole_size/2)::]).reshape(-1, self.seg_num, self.joint_num, 1, int(self.hidden1/2))), dim=4)

		CELL0 = torch.cat((encoded_whole, encoded_diff, encoded_Comm0), dim=3)
		
		if self.mode == 'identity':
			CELL0 = torch.cat((encoded_whole, encoded_diff, encoded_Comm0), dim=3)
			CELL3 = torch.cat((encoded_whole, encoded_diff, encoded_Comm0), dim=3)
		else:
			CELL0 = torch.cat((encoded_whole, encoded_diff, encoded_Comm0, \
				encoded_Comm1_Con1, encoded_Comm1_Con2, encoded_Comm1_Con3, \
				encoded_Diff0_Con1, encoded_Diff0_Con2, encoded_Diff0_Con3, \
				encoded_Diff1_Con1, encoded_Diff1_Con2, encoded_Diff1_Con3), dim=3)
			Cube01_rgb, Cube01_flow = self.get_Nas(CELL0[:,:,:,:,0:int(self.hidden1/2)], CELL0[:,:,:,:,int(self.hidden1/2)::], self.conv01_max_joint_rgb, self.conv01_max_module_rgb, \
				self.FC_Reduce01_joint_rgb, self.FC_Reduce01_module_rgb, self.conv01_Red_joint_rgb, self.conv01_Red_module_rgb,\
				self.conv01_max_time_rgb, self.FC_Reduce01_time_rgb, self.conv01_Red_time_rgb, self.attention_01_rgb, self.conv3d01_rgb, \
				self.conv01_max_joint_flow, self.conv01_max_module_flow, \
				self.FC_Reduce01_joint_flow, self.FC_Reduce01_module_flow, self.conv01_Red_joint_flow, self.conv01_Red_module_flow,\
				self.conv01_max_time_flow, self.FC_Reduce01_time_flow, self.conv01_Red_time_flow, self.attention_01_flow, self.conv3d01_flow, self.operation_selector01)

			Cube02_rgb, Cube02_flow = self.get_Nas(CELL0[:,:,:,:,0:int(self.hidden1/2)], CELL0[:,:,:,:,int(self.hidden1/2)::], self.conv02_max_joint_rgb, self.conv02_max_module_rgb, \
				self.FC_Reduce02_joint_rgb, self.FC_Reduce02_module_rgb, self.conv02_Red_joint_rgb, self.conv02_Red_module_rgb,\
				self.conv02_max_time_rgb, self.FC_Reduce02_time_rgb, self.conv02_Red_time_rgb, self.attention_02_rgb, self.conv3d02_rgb,\
				self.conv02_max_joint_flow, self.conv02_max_module_flow, \
				self.FC_Reduce02_joint_flow, self.FC_Reduce02_module_flow, self.conv02_Red_joint_flow, self.conv02_Red_module_flow,\
				self.conv02_max_time_flow, self.FC_Reduce02_time_flow, self.conv02_Red_time_flow, self.attention_02_flow, self.conv3d02_flow, self.operation_selector02)

			Cube03_rgb, Cube03_flow = self.get_Nas(CELL0[:,:,:,:,0:int(self.hidden1/2)], CELL0[:,:,:,:,int(self.hidden1/2)::], self.conv03_max_joint_rgb, self.conv03_max_module_rgb, \
				self.FC_Reduce03_joint_rgb, self.FC_Reduce03_module_rgb,  self.conv03_Red_joint_rgb, self.conv03_Red_module_rgb,\
				self.conv03_max_time_rgb, self.FC_Reduce03_time_rgb, self.conv03_Red_time_rgb, self.attention_03_rgb, self.conv3d03_rgb,\
				self.conv03_max_joint_flow, self.conv03_max_module_flow, \
				self.FC_Reduce03_joint_flow, self.FC_Reduce03_module_flow,  self.conv03_Red_joint_flow, self.conv03_Red_module_flow,\
				self.conv03_max_time_flow, self.FC_Reduce03_time_flow, self.conv03_Red_time_flow, self.attention_03_flow, self.conv3d03_flow, self.operation_selector03)

	
			CELL1 = torch.cat((Cube01_rgb, Cube01_flow), dim=4) 
	
			Cube12_rgb, Cube12_flow = self.get_Nas(CELL1[:,:,:,:,0:int(self.hidden1/2)], CELL1[:,:,:,:,int(self.hidden1/2)::], self.conv12_max_joint_rgb, self.conv12_max_module_rgb, \
				self.FC_Reduce12_joint_rgb, self.FC_Reduce12_module_rgb, self.conv12_Red_joint_rgb, self.conv12_Red_module_rgb,\
				self.conv01_max_time_rgb, self.FC_Reduce12_time_rgb, self.conv12_Red_time_rgb, self.attention_12_rgb, self.conv3d12_rgb,\
				self.conv12_max_joint_flow, self.conv12_max_module_flow, \
				self.FC_Reduce12_joint_flow, self.FC_Reduce12_module_flow, self.conv12_Red_joint_flow, self.conv12_Red_module_flow,\
				self.conv01_max_time_flow, self.FC_Reduce12_time_flow, self.conv12_Red_time_flow, self.attention_12_flow, self.conv3d12_flow, self.operation_selector12)

			Cube13_rgb, Cube13_flow = self.get_Nas(CELL1[:,:,:,:,0:int(self.hidden1/2)], CELL1[:,:,:,:,int(self.hidden1/2)::], self.conv13_max_joint_rgb, self.conv13_max_module_rgb, \
				self.FC_Reduce13_joint_rgb, self.FC_Reduce13_module_rgb,  self.conv13_Red_joint_rgb, self.conv13_Red_module_rgb, \
				self.conv01_max_time_rgb, self.FC_Reduce13_time_rgb, self.conv13_Red_time_rgb, self.attention_13_rgb, self.conv3d13_rgb,\
				self.conv13_max_joint_flow, self.conv13_max_module_flow, \
				self.FC_Reduce13_joint_flow, self.FC_Reduce13_module_flow,  self.conv13_Red_joint_flow, self.conv13_Red_module_flow,\
				self.conv01_max_time_flow, self.FC_Reduce13_time_flow, self.conv13_Red_time_flow, self.attention_13_flow, self.conv3d13_flow, self.operation_selector13)

	
			CELL2 = torch.cat((Cube02_rgb+Cube12_rgb, Cube02_flow+Cube12_flow), dim=4) 
	
			Cube23_rgb, Cube23_flow = self.get_Nas(CELL2[:,:,:,:,0:int(self.hidden1/2)], CELL2[:,:,:,:,int(self.hidden1/2)::], self.conv23_max_joint_rgb, self.conv23_max_module_rgb, \
				self.FC_Reduce23_joint_rgb, self.FC_Reduce23_module_rgb, self.conv23_Red_joint_rgb, self.conv23_Red_module_rgb,\
				self.conv23_max_time_rgb, self.FC_Reduce23_time_rgb, self.conv23_Red_time_rgb, self.attention_23_rgb, self.conv3d23_rgb,\
				self.conv23_max_joint_flow, self.conv23_max_module_flow, \
				self.FC_Reduce23_joint_flow, self.FC_Reduce23_module_flow, self.conv23_Red_joint_flow, self.conv23_Red_module_flow,\
				self.conv23_max_time_flow, self.FC_Reduce23_time_flow, self.conv23_Red_time_flow, self.attention_23_flow, self.conv3d23_flow, self.operation_selector23)
	 
			CELL3 = torch.cat((Cube03_rgb+Cube13_rgb+Cube23_rgb, Cube03_flow+Cube13_flow+Cube23_flow), dim=4) 

		
		cosine_tensor = torch.abs(F.cosine_similarity(encoded_whole, encoded_Comm0, dim=4))\
			+torch.abs(F.cosine_similarity(encoded_whole, encoded_Comm1_Con1, dim=4))\
			+torch.abs(F.cosine_similarity(encoded_whole, encoded_Comm1_Con2, dim=4))\
			+torch.abs(F.cosine_similarity(encoded_whole, encoded_Comm1_Con3, dim=4))\
			+torch.abs(F.cosine_similarity(encoded_whole, encoded_Diff0_Con1, dim=4))\
			+torch.abs(F.cosine_similarity(encoded_whole, encoded_Diff0_Con2, dim=4))\
			+torch.abs(F.cosine_similarity(encoded_whole, encoded_Diff0_Con3, dim=4))\
			+torch.abs(F.cosine_similarity(encoded_whole, encoded_Diff1_Con1, dim=4))\
			+torch.abs(F.cosine_similarity(encoded_whole, encoded_Diff1_Con2, dim=4))\
			+torch.abs(F.cosine_similarity(encoded_whole, encoded_Diff1_Con3, dim=4))
		# Fusing
		fused_feat = torch.mean(torch.mean(torch.mean(torch.cat((CELL0[:,:,:,:,0:int(self.hidden1/2)]+CELL0[:,:,:,:,int(self.hidden1/2)::],\
		CELL3[:,:,:,:,0:int(self.hidden1/2)]+CELL3[:,:,:,:,int(self.hidden1/2)::]), dim=4), dim=3), dim=2), dim=1) #torch.mean(torch.mean(torch.mean(CELL0, dim=3), dim=2), dim=1)
		
		tot_scores = self.assessment2(self.assessment1(fused_feat)).reshape(-1)
		return tot_scores, None, None, None, None, None, None, fused_feat
		
def get_numpy_mse(pred, score):
	pred = np.array(pred)
	score = np.array(score)
	return np.sum((pred - score)**2) / pred.shape[0]

def get_numpy_spearman(pred, score):
	pred = np.array(pred)
	score = np.array(score)
	import pdb
	#pdb.set_trace()
	return stats.spearmanr(pred, score).correlation

def get_numpy_pearson(pred, score):
	pred = np.array(pred)
	score = np.array(score)
	return stats.pearsonr(pred, score)[0]



