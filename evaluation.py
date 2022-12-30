import numpy as np
import sys
import scipy.io
import pdb
import os
import scipy.stats as stats
from sklearn import svm
from sklearn import metrics
import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as Ff
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
import argparse
from torch.nn import init
from torch.optim.lr_scheduler import *
from model_JRG10_com5_v8 import *
import time
import shutil
import random

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('-gpu', type=str)
parser.add_argument('-time', type=int, default=0, choices=[0,1,2,3,4,5,6,7,8,9])
parser.add_argument('-mode', type=str, \
	choices=['base', 'base2', 'choice', 'structure','random','identity'])
parser.add_argument('-action-set', type=int, default=0, choices=[0,1,2])
parser.add_argument('-loss-type', type=str, default='mse', \
	choices=['mse', 'new_mse', 'pearson', 'sigmoid'])
parser.add_argument('-cyclic', type=int, default=0, choices=[0,1], help='0 for none-cyclic, 1 for cyclic')
parser.add_argument('-set-id', type=int, default=1, choices=[0,1,2,3,4,5], help='Dataset id(0-3)')
parser.add_argument('-batch-size', type=int, default=48)
parser.add_argument('-lr', type=float, default=1e-5, help='Initial Learning Rate')
parser.add_argument('-tf-or-torch', type=str, default='torch')
parser.add_argument('-kinetics-or-charades', type=str, default='kinetics')
parser.add_argument('-swind-or-segm', type=str, default='swind')

parser.add_argument('--max-epoch', type=int, default=5000, \
	help='Max Epoch for Pre-Training the encoders and scoring module')
parser.add_argument('--seg-num', type=int, default=12, help='Number of Video Segments')
parser.add_argument('--joint-num', type=int, default=17, help='Number of human joints')
parser.add_argument('--whole-size', type=int, default=400, help='I3D feat size')
parser.add_argument('--patch-size', type=int, default=400, help='I3D feat size')

parser.add_argument('--sigma', type=float, default=0)
parser.add_argument('--margin', type=float, default=1e-3)
args = parser.parse_args()

args.mode = 'structure'
args.batch_size=int(2)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

act_names = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']

structure_rootdir = './checkpooints/trained/210517-AQA7-structure-v8_1'
print (structure_rootdir)
print_frequency = 20
accumulate_step = 0
epoch = 0

sample_scores = None

def run_assessment(model_, optim_, loader, is_train, is_print= False):
	global accumulate_step, epoch, print_frequency, start
	if is_train:
		model_.train()
	else:
		model_.eval()
	loss_ = torch.tensor(0.0).cuda()
	mse_loss_ = torch.tensor(0.0).cuda()
	orth_loss_ = torch.tensor(0.0).cuda()
	l2_loss_ = torch.tensor(0.0).cuda()
	sabs_loss_ = torch.tensor(0.0).cuda()
	pred_vec = torch.tensor([]).cuda()
	grou_vec = torch.tensor([]).cuda()
	for step, (feat_whole, feat_patch, scores, _) in enumerate(loader):
		if is_train:
			accumulate_step += 1
			if accumulate_step == 1:
				epoch += 1
				accumulate_step = 0
			if epoch < 100 and epoch % 10 == 0 and accumulate_step == 0:
				print ('Time before epoch', epoch, ':', time.time()-start)
				start = time.time()
		feat_whole = feat_whole.cuda()
		feat_patch = feat_patch.cuda()
		scores = scores.float().cuda()
		pred, _, _, _, cosine_tensor, l2_tensor, _, _ = model_(args.loss_type, feat_whole, feat_patch, scores)
		mse_loss = get_loss(pred, scores, args.loss_type)
		loss = mse_loss
		if is_train:
			loss.backward(retain_graph=True)
			if accumulate_step == 0:
				optim_.step()
				optim_.zero_grad()
		loss_+= loss.detach()
		mse_loss_ += mse_loss.detach()
		if step == 0:
			pred_vec = pred.detach().reshape(-1)#torch.max(pred, dim=1)[1].float().detach().reshape(-1)#
			grou_vec = scores.detach().reshape(-1)
		else:
			pred_vec = torch.cat((pred_vec, pred.detach().reshape(-1)), dim=0)
			grou_vec = torch.cat((grou_vec, scores.detach().reshape(-1)), dim=0)
		if is_train and epoch % print_frequency == 0 and accumulate_step == 0:
			is_print = True
			break

	if is_print:
		mse = get_numpy_mse(pred_vec.data.cpu().numpy(), grou_vec.data.cpu().numpy())
		spearman = get_numpy_spearman(pred_vec.data.cpu().numpy(), grou_vec.data.cpu().numpy())
		pearson = get_numpy_pearson(pred_vec.data.cpu().numpy(), grou_vec.data.cpu().numpy())
		avg_loss = loss_.data.cpu().numpy() / len(loader) 
		avg_mse = mse_loss_.data.cpu().numpy() / len(loader) 
		avg_orth = orth_loss_.data.cpu().numpy() / len(loader) 
		avg_l2 = l2_loss_.data.cpu().numpy() / len(loader)
		avg_sabs = sabs_loss_.data.cpu().numpy() / len(loader)
		sample_num = len(pred_vec)
		return model_, optim_, sample_num, mse, spearman, pearson, avg_loss, avg_mse, avg_orth, avg_l2, avg_sabs
	else:
		return model_, optim_, None, None, None, None, None, None, None, None, None

train_whole_vec = []
test_whole_vec = []
train_patch_vec = []
test_patch_vec = []
train_scores_vec = []
test_scores_vec = []

# Import Dataset
start_time = time.time()
feat_file = os.path.join('features', 'AQA_pytorch_'+args.kinetics_or_charades+'_'+args.swind_or_segm+'_Set_'+str(args.set_id+1)+'_Feats.npz')
print ('Loaded feature file:', feat_file)
all_dict = np.load(feat_file)
test_whole = np.concatenate((all_dict['test_rgb'][:,:,0,:],all_dict['test_flow'][:,:,0,:]), axis=2)
test_patch = np.concatenate((all_dict['test_rgb'][:,:,1::,:].transpose((0,2,1,3)),all_dict['test_flow'][:,:,1::,:].transpose((0,2,1,3))), axis=3)
train_whole = np.concatenate((all_dict['train_rgb'][:,:,0,:], all_dict['train_flow'][:,:,0,:]), axis=2)
train_patch = np.concatenate((all_dict['train_rgb'][:,:,1::,:].transpose((0,2,1,3)),all_dict['train_flow'][:,:,1::,:].transpose((0,2,1,3))), axis=3)

train_scores_ = all_dict['train_label']
train_scores_ += np.random.normal(0,args.sigma, train_scores_.shape[0]) * (np.max(np.array(train_scores_)) - np.min(np.array(train_scores_)))
train_scores = np.repeat(train_scores_, 2)
test_scores = all_dict['test_label']
args.seg_num = test_whole.shape[1]
args.feat_size = test_whole.shape[2]
args.whole_size = args.patch_size = args.feat_size

# Score Normalize
train_max =  np.max(train_scores)
train_min = np.min(train_scores)
train_scores = (train_scores - train_min) / (train_max - train_min) * 10.0
test_scores = (test_scores - train_min) / (train_max - train_min) * 10.0

test_scores_vec += test_scores.tolist()
test_whole_vec += test_whole.tolist()
test_patch_vec += test_patch.tolist()

dataset_test = Data.TensorDataset( \
					torch.tensor(test_whole).float(), \
					torch.tensor(test_patch).float(), \
					torch.tensor(test_scores).float(),
					torch.tensor(np.array([args.set_id for j in range(test_scores.shape[0])])))
dataset_train = Data.TensorDataset( \
					torch.tensor(train_whole).float(), \
					torch.tensor(train_patch).float(), \
					torch.tensor(train_scores).float(),
					torch.tensor(np.array([args.set_id for j in range(train_scores.shape[0])])))

loader_test = Data.DataLoader(\
	dataset=dataset_test, batch_size=args.batch_size, \
	shuffle=False, num_workers=2)

loader_train = Data.DataLoader(\
	dataset=dataset_train, batch_size=args.batch_size, \
	shuffle=False, num_workers=2)

print ('dataset making time cost: ', time.time()-start_time)

# Start Evaluation
print ('Action', act_names[args.set_id])
model_ = ASS_JRG(whole_size=args.whole_size, patch_size=args.patch_size, seg_num=args.seg_num, joint_num=args.joint_num, mode = args.mode)
checkpoint_path = os.path.join('./release/checkpoints/AQA7/','set_'+str(act_names[args.set_id])+'_mode_'+args.mode+'_loss_'+args.loss_type+'.pt')
model_.load_state_dict(torch.load(checkpoint_path))
model_ = nn.DataParallel(model_, device_ids=[0])

print ("operation_selector01.weight", model_.module.operation_selector01.weight.detach())
print ("operation_selector02.weight", model_.module.operation_selector02.weight.detach())
print ("operation_selector03.weight", model_.module.operation_selector03.weight.detach())
print ("operation_selector12.weight", model_.module.operation_selector12.weight.detach())
print ("operation_selector13.weight", model_.module.operation_selector13.weight.detach())
print ("operation_selector23.weight", model_.module.operation_selector23.weight.detach())

start = time.time()
epoch = print_frequency
model_.eval()
model_, _, sample_num_test, \
	mse_test, spearman_test, pearson_test, \
	avg_loss_test, avg_pearson_test, avg_cosine_test, avg_l2_test, avg_sabs_test = \
	run_assessment(model_, None, loader_test, False, is_print=True)
print ('Spear. Corr. in Test set: %.4f' % spearman_test)