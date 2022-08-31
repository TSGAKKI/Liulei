import numpy as np
import os
import pickle
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import utils
from data_prepare import *
from args import get_args
from model import *
from json import dumps
# from tensorboardX import SummaryWriter

from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

#t-rrmse
def denoise_loss_mse(denoise, clean):      
    loss = (denoise-clean)**2
    return torch.mean(loss)
def denoise_loss_rmse(denoise, clean):      #tmse
    loss = (denoise-clean)**2
    return torch.sqrt(torch.mean(loss))
def denoise_loss_rrmset(denoise, clean):      #tmse
  
    # print(denoise.shape) #  3400 512
    # print(tf.zeros(clean.shape, tf.float64).shape)
    # exit(0)
    rmse1 = denoise_loss_rmse(denoise, clean)
    rmse2 = denoise_loss_rmse(clean, torch.zeros_like(clean).to(clean.device))
    #loss2 = tf.losses.mean_squared_error(noise, clean)
    return rmse1/rmse2

def get_corr(pred, label):#计算两个向量person相关系数
    # pred # BT
    # label # BT
   
    pred_mean, label_mean = torch.mean(pred,dim=-1,keepdim=True), torch.mean(label,dim=-1,keepdim=True)
    
    corr = ( torch.mean((pred - pred_mean) * (label - label_mean), dim=-1, keepdim=True) ) / (
                torch.sqrt(torch.mean((pred - pred_mean) ** 2, dim=-1, keepdim=True)) * torch.sqrt(torch.mean((label - label_mean) ** 2, dim=-1, keepdim=True )))
    
    return torch.mean(corr)


def main(args):
    # Get device
    # args.cuda = torch.cuda.is_available()
    device = torch.device(  'cuda:{}'.format(args.cuda)  )
    # Set random seed
    utils.seed_torch(seed = args.rand_seed)
    # Get save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, training=True if args.do_train else False)
    # Save superpara: args
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
    # Set up logger
    log = utils.get_logger(args.save_dir, 'train')
    # tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    # Build dataset
    log.info('Building dataset...')
    if args.noise_type == 'EOG':
        EEG_all = np.load( args.data_dir + 'EEG_all_epochs.npy')                              
        noise_all = np.load( args.data_dir + 'EOG_all_epochs.npy') 
    elif args.noise_type == 'EMG':
        EEG_all = np.load( args.data_dir + 'EEG_all_epochs_512hz.npy')                              
        noise_all = np.load( args.data_dir + 'EMG_all_epochs_512hz.npy')
    print('EEG_all:',EEG_all.shape) # 4514 512
    print('noise_all:',noise_all.shape) # 3400 512
   
    # GPU 
    noiseEEG_train_end_standard, EEG_train_end_standard, noiseEEG_test_end_standard, EEG_test_end_standard, std_VALUE = prepare_data(device, args.batch_size, EEG_all = EEG_all, noise_all = noise_all, combin_num = 10, \
        train_num = 3000, noise_type = args.noise_type)
    
        #--input data
    train_x = torch.from_numpy(noiseEEG_train_end_standard).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, F, T)
    train_target = torch.from_numpy(EEG_train_end_standard).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
    train_dataset = torch.utils.data.TensorDataset(train_x, train_target)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    test_x = torch.from_numpy(noiseEEG_test_end_standard).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, F, T)
    test_target = torch.from_numpy(EEG_test_end_standard).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_target)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    #k    
    # k=10
    # all_scores=[]
    # num_val_samples=len(noiseEEG_train_end_standard)//k
    # print('num_val_samples',num_val_samples)
    # for i in range(10):
    #     log.info('processing fold # {}'.format(i))
    #     partial_noiseEEG_train=np.concatenate([noiseEEG_train_end_standard[:i*num_val_samples], noiseEEG_train_end_standard[(i+1)*num_val_samples:]], axis=0) 
    #     partial_EEG_train=np.concatenate([EEG_train_end_standard[:i*num_val_samples],EEG_train_end_standard[(i+1)*num_val_samples:]],axis=0) 
    #     print('partial_noiseEEG_train:',np.shape(partial_noiseEEG_train))
        
    #     noiseEEG_val=noiseEEG_train_end_standard[i*num_val_samples:(i+1)* num_val_samples]
    #     EEG_val=EEG_train_end_standard[i*num_val_samples:(i+1)* num_val_samples]
    #     print('noiseEEG_val:',np.shape(noiseEEG_val))
    #     print('EEG_val:',np.shape(EEG_val)) 
  

    #     val_x = torch.from_numpy(noiseEEG_val).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, F, T)
    #     val_target = torch.from_numpy(EEG_val).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)
    #     val_dataset = torch.utils.data.TensorDataset(val_x, val_target)
    #     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
    # Build model
    log.info('Building model...')
    model = make_model(
        args=args, DEVICE=device)

    if args.do_train:
        if args.load_model_path is not None:
            model = utils.load_model_checkpoint(
                args.load_model_path, model)
    
        num_params = utils.count_parameters(model)
        log.info('Total number of trainable parameters: {}'.format(num_params))

        # Train
        train(model, train_loader, args, device, args.save_dir, log)#, tbx

        # Load best model after training finished
        best_path = os.path.join(args.save_dir, 'best.pth.tar')
        model = utils.load_model_checkpoint(best_path, model)
        model = model.to(device)
    elif ( not args.do_train ) and args.load_model_path is None:
        raise ValueError(
                'For fine-tuning, provide pretrained model in load_model_path!')

    # Evaluate on test set
    # val_results = evaluate(model,
    #                         val_loader,
    #                         args,
    #                         device,
    #                         args.save_dir,log,                          
    #                         is_test=False
    #                     )
    test_results = evaluate(model,
                        test_loader,
                        args,
                        device,
                        args.save_dir,log,                          
                        is_test=True,
                        )
    # Log to console

#        test_results_str = ', '.join('{}: {:.3f}'.format(k, v)
#                                    for k, v in val_results.items())
#        log.info('TEST set prediction results: {}'.format(test_results_str))


def train(model, train_loader,args, device, save_dir, log): #, tbx
    """
    Perform training and evaluate on val set
    """
    # Get saver
    saver = utils.CheckpointSaver(save_dir,log=log)

    # To train mode
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr_init)

    # Train
    log.info('Training...')
    epoch = 0
    
    prev_val_loss = 1e10
    patience_count = 0
    early_stop = False
    while (epoch != args.num_epochs) and (not early_stop):
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        total_samples = len(train_loader)
        for noiseeeg_batch, cleaneeg_batch in train_loader:
            noiseeeg_batch = noiseeeg_batch.to(device)
            cleaneeg_batch = cleaneeg_batch.to(device)
            
            denoiseoutput = model(noiseeeg_batch)
           
            loss = denoise_loss_mse(denoiseoutput, cleaneeg_batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #     # Back to train mode
        #     model.train()

        #     # Log to console
        #     # results_str = ', '.join('{}: {:.3f}'.format(k, v)
        #     #                         for k, v in eval_results.items())
        #     # log.info('Dev {}'.format(results_str))

        #     # Log to TensorBoard
        #     # log.info('Visualizing in TensorBoard...')
        #     # for k, v in eval_results.items():
        #     #     tbx.add_scalar('eval/{}'.format(k), v, step)


def evaluate(model,dataloader,args,DEVICE,save_dir,log,is_test=False):
    # To evaluate mode
    model.eval()

    val_losses = []
    y_pred_all = []
    y_true_all = []
    for noiseeeg_batch, cleaneeg_batch in dataloader:
        noiseeeg_batch = noiseeeg_batch.to(DEVICE)
        cleaneeg_batch = cleaneeg_batch.to(DEVICE)
        denoiseoutput = model(noiseeeg_batch)
        # Update loss
        loss = denoise_loss_mse(denoiseoutput, cleaneeg_batch)
        val_losses.append(loss.item())
        y_pred_all.append(denoiseoutput)
        y_true_all.append(cleaneeg_batch)
      
    validation_loss = sum(val_losses) / len(val_losses)

    if is_test:
       
        y_pred_all = torch.cat( y_pred_all,dim=0)
        y_true_all = torch.cat( y_true_all,dim=0)

        rrmset = denoise_loss_rrmset(y_pred_all, y_true_all)
        cc = get_corr(y_pred_all, y_true_all)
        log.info('Test result: test_rrmset:{:.3f}, test_corr:{:.3f}'.format( rrmset, cc))
    else:
        log.info('val result: validation_loss:{:.3f}'.format( validation_loss))
    return validation_loss

if __name__ == '__main__':
    main(get_args())
