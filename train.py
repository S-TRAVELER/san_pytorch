#!/usr/bin/env python

import torch
from torch import nn
from torch.autograd import Variable
import argparse
import numpy
import logging
import log

from san import LSTM_Att
from data_provision_att_vqa import *
from data_processing_vqa import *


##################
# initialization #
##################
options = OrderedDict()
# data related
options['data_path'] = '/home/data'
options['feature_file'] = 'trainval_feat.h5'
options['expt_folder'] = 'expt_1'
options['model_name'] = 'imageqa'
options['train_split'] = 'trainval1'
options['val_split'] = 'val2'
options['shuffle'] = True
options['reverse'] = True
options['sample_answer'] = True

options['num_region'] = 196
options['region_dim'] = 512

options['n_words'] = 13746
options['n_output'] = 1000

# structure options
options['combined_num_mlp'] = 1
options['combined_mlp_drop_0'] = True
options['combined_mlp_act_0'] = 'linear'
options['sent_drop'] = False
options['use_tanh'] = False

options['use_attention_drop'] = False

# dimensions
options['n_emb'] = 500
options['n_dim'] = 1024
options['n_image_feat'] = options['region_dim']
options['n_common_feat'] = 500
options['n_attention'] = 512

# initialization
options['init_type'] = 'uniform'
options['range'] = 0.01
options['std'] = 0.01
options['init_lstm_svd'] = False

options['forget_bias'] = numpy.float32(1.0)

# learning parameters
options['optimization'] = 'sgd' # choices
options['batch_size'] = 100
options['lr'] = numpy.float32(0.05)
options['w_emb_lr'] = numpy.float32(80)
options['momentum'] = numpy.float32(0.9)
options['gamma'] = 1
options['step'] = 10
options['step_start'] = 100
options['max_epochs'] = 50
options['weight_decay'] = 0.0005
options['decay_rate'] = numpy.float32(0.999)
options['drop_ratio'] = numpy.float32(0.5)
options['smooth'] = numpy.float32(1e-8)
options['grad_clip'] = numpy.float32(0.1)

# log params
options['disp_interval'] = 10
options['eval_interval'] = 1000
options['save_interval'] = 500

def train(options):
    logger = logging.getLogger('root')
    logger.info(options)
    logger.info('start training')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(device)
    
    att_lstm_net=LSTM_Att(options).to(device) 
    
    logger.info('finished building model')
    #print(att_lstm_net)
    loss_func =nn.CrossEntropyLoss()
    #loss_func=MLoss()
   
    optimizer = torch.optim.SGD(att_lstm_net.parameters(),lr=options['lr'], momentum=options['momentum'],weight_decay=options['weight_decay'])
    #optimizer = torch.optim.RMSprop(att_lstm_net.parameters(),lr=options['lr'], momentum=options['momentum'])
    #optimizer = torch.optim.Adagrad(att_lstm_net.parameters(), weight_decay=options['weight_decay'])
    
    #                            weight_decay=options['weight_decay'] )
    # optimizer = torch.optim.Adam(att_lstm_net.parameters(), lr=options['lr'], betas=(0.9, 0.99), weight_decay=options['weight_decay'])

    batch_size = options['batch_size']
    max_epochs = options['max_epochs']

    data_provision_att_vqa = DataProvisionAttVqa(options['data_path'],options['feature_file'])

    num_iters_one_epoch = data_provision_att_vqa.get_size(options['train_split']) // batch_size
    max_iters = max_epochs * num_iters_one_epoch
    
    eval_interval_in_iters = options['eval_interval']
    save_interval_in_iters = options['save_interval']
    disp_interval = options['disp_interval']
    logger.info('finished building function')
    
    # training and testing
    for iter in range(max_iters + 1):
        att_lstm_net.setdrop(True)
        if options['sample_answer']:
            batch_image_feat, batch_question, batch_answer_label \
                = data_provision_att_vqa.next_batch_sample(options['train_split'],
                                                       batch_size)
        else:
            batch_image_feat, batch_question, batch_answer_label \
                = data_provision_att_vqa.next_batch(options['train_split'], batch_size)
        input_idx, input_mask \
            = process_batch_l(batch_question, reverse=options['reverse'])
            
        batch_image_feat = reshape_image_feat(batch_image_feat,
                                            options['num_region'],
                                            options['region_dim'])

        

        batch_answer_label=batch_answer_label.astype('int32').flatten()

        batch_image_feat=Variable(torch.FloatTensor(batch_image_feat).to(device))
        input_idx=Variable(torch.LongTensor(input_idx).to(device))
        batch_answer_label=Variable(torch.LongTensor(batch_answer_label).to(device),requires_grad=False)
        input_mask=Variable(torch.Tensor(input_mask).to(device),requires_grad=False)
            
        #batch_image_feat=nn.functional.normalize(batch_image_feat,dim=2)
                                  # clear gradients for this training step
        input_idx=torch.transpose(input_idx,1,0)
        input_mask=torch.transpose(input_mask,1,0)
        
       
        
        pred_label = att_lstm_net(batch_image_feat, input_idx, input_mask)                     # rnn output
        loss = loss_func(pred_label, batch_answer_label) 
        
        
        reg_cost=att_lstm_net.cost_weights()*options['weight_decay'] 
       
        optimizer.zero_grad() 
        #loss+=reg_cost
        
        loss.backward()                                 # backpropagation, compute gradients
        #梯度裁剪
        torch.nn.utils.clip_grad_norm_(att_lstm_net.parameters(), options['grad_clip'])
        
        optimizer.step()                                # apply gradients
      
        if options['shuffle'] and iter > 0 and iter % num_iters_one_epoch == 0:
            data_provision_att_vqa.random_shuffle()
            
        if (iter % disp_interval) == 0  or (iter == max_iters):
            accu = torch.mean(torch.max(pred_label, 1)[1].data.eq(batch_answer_label).float())
            loss_d = loss.data.cpu().numpy()
            #print(reg_cost)
            #print(torch.std(batch_image_feat, dim=(1, 2)))
            #print(torch.mean(batch_image_feat, dim=(1, 2)))
            logger.info('iteration %d/%d epoch %f/%d loss %f accu %f' \
                        % (iter, max_iters,
                           iter / float(num_iters_one_epoch), max_epochs,
                           loss_d, accu))
        
        if iter > 0 and ((iter % eval_interval_in_iters) == 0 or (iter == max_iters)):
            val_loss_list = []
            val_accu_list = []
            val_count = 0
            
            att_lstm_net.setdrop(False)
            
            for batch_image_feat, batch_question, batch_answer_label \
                in data_provision_att_vqa.iterate_batch(options['val_split'],
                                                    batch_size):
                input_idx, input_mask \
                    = process_batch_l(batch_question,
                                    reverse=options['reverse'])
                    
                batch_answer_label=batch_answer_label.astype('int32').flatten()
                batch_image_feat = reshape_image_feat(batch_image_feat,
                                                    options['num_region'],
                                                    options['region_dim'])
                
                        
                batch_image_feat=Variable(torch.FloatTensor(batch_image_feat).to(device))
                input_idx=Variable(torch.LongTensor(input_idx).to(device))
                batch_answer_label=Variable(torch.LongTensor(batch_answer_label).to(device),requires_grad=False)
                input_mask=Variable(torch.Tensor(input_mask).to(device),requires_grad=False)
                
                #batch_image_feat=nn.functional.normalize(batch_image_feat,dim=2)
                
                input_idx=torch.transpose(input_idx,1,0)
                input_mask=torch.transpose(input_mask,1,0)
                
                pred_label = att_lstm_net(batch_image_feat, input_idx, input_mask)                        # rnn output
                loss = loss_func(pred_label, batch_answer_label) 
                
                
                accu = torch.mean(torch.max(pred_label, 1)[1].data.eq(batch_answer_label).float())
                val_count += batch_image_feat.shape[0]
                
                val_accu_list.append(accu * batch_image_feat.shape[0])
                val_loss_list.append(loss.data.cpu().numpy()* batch_image_feat.shape[0])
               
                
            ave_val_accu = sum(val_accu_list) / float(val_count)
            ave_val_loss = sum(val_loss_list) / float(val_count)
            logger.info('validation loss: %f accu: %f '%(ave_val_loss, ave_val_accu ))




if __name__ == '__main__':
    logger = log.setup_custom_logger('root')
    parser = argparse.ArgumentParser()
    parser.add_argument('changes', nargs='*',
                        help='Changes to default values',
                        default = '')
    args = parser.parse_args()
    for change in args.changes:
        logger.info('dict({%s})'%(change))
        options.update(eval('dict({%s})'%(change)))
    train(options)