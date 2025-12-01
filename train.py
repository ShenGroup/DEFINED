import os
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from torch.optim.lr_scheduler import LambdaLR
from transformers import (AdamW, GPT2Config, GPT2Model,
                          get_linear_schedule_with_warmup)
import wandb
from data import (count_modulation_symbols, generate_signals)
from parameters import parameter_reading


def build_model(embedding_dim, n_positions,num_heads, num_layers, data_dim, num_classes):
    model = TransformerModel(
        n_dims=data_dim,
        n_positions=2 *n_positions,
        n_embd=embedding_dim,
        n_layer=num_layers,
        n_head=num_heads,
        n_classes=num_classes
    )
    return model


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd, n_layer, n_head, n_classes):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_classes = n_classes
        self._read_in = nn.Linear(self.n_classes, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, self.n_classes)

    @staticmethod
    def _combine(ys_b,xs_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        _, _, dim_y = ys_b.shape
        if dim_y < dim:
            padding_size = dim - dim_y
            padding = torch.zeros(bsize, points, padding_size, device=ys_b.device, dtype=ys_b.dtype)
            ys_b = torch.cat((ys_b, padding), dim=-1)
        zs = torch.stack((ys_b,xs_b), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs
    def forward(self, ys_batch, xs_batch, inds=None):
        if inds is None:
            inds = torch.arange(xs_batch.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= xs_batch.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(ys_batch, xs_batch)
        zs = zs.to(torch.float32)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        '''Mapping to Constallation Symbol'''
        # prediction = (torch.sigmoid(prediction)-0.5)*np.sqrt(2)
        bsize, points, dim = ys_batch.shape
        return prediction[:, ::2, :]


def train_step(model, ys_batch, xs_batch, xs_real, optimizer, loss_func):
    model.train()
    x_prob = model(ys_batch, xs_batch)
    xs_real_indices = torch.argmax(xs_real, dim=-1)
    loss = loss_func(x_prob.transpose(1, 2), xs_real_indices).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()    # 更新学习率
    return loss.detach().item(), x_prob.detach()


def sequence_train_step(args,model, ys_batch, xs_batch, xs_real, optimizer, loss_func, train_pilot_len,
                        label_encoder):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    bsize, length, dim = xs_batch.shape
    xin = torch.zeros_like(xs_batch)
    yin = torch.zeros_like(xs_batch)
    seq_loss = []
    total_loss = 0
    pilot_len = train_pilot_len

    with torch.no_grad():
        for i in range(length):
            if i < pilot_len:
                xin = torch.cat([xin[:, :i, :], xs_batch[:, i:, :]], dim=1)
            else:
                x_hat = model(ys_batch, xin)
                probabilities = torch.softmax(x_hat, dim=-1)
                _, max_indices = torch.max(probabilities, dim=-1)
                one_hot_encoded = torch.nn.functional.one_hot(max_indices, num_classes=args.modu_num)
                xin = torch.cat([xin[:, :i, :], one_hot_encoded[:, i:, :]], dim=1)

    x_prob1 = model(ys_batch, xin)
    xs_real_indices = torch.argmax(xs_real, dim=-1)
    loss1 = loss_func(x_prob1.transpose(1, 2), xs_real_indices) #  [batch_size, sequence_length]



    x_prob2 = model(ys_batch, xs_batch)
    xs_real_indices = torch.argmax(xs_real, dim=-1)
    loss2 = loss_func(x_prob2.transpose(1, 2), xs_real_indices)

    weight = args.loss_weight
    total_loss = ( weight*loss1 + (1-weight)*loss2).mean()

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.detach(), x_prob1.detach()



def complex_to_vec(X):
    "Converts complex matrix to real vector"
    X_vec = np.concatenate([np.real(X) , np.imag(X) ], axis=2)
    return X_vec



def trainNetwork(model_GPT2, args, n_tasks, task_name):

    YOUR_API_KEY = '8671afdd4e4352912e7dbf1b2abe5f5fb2f253a2'
    wandb.login(key = YOUR_API_KEY)
    wandb.init(
        project='SISO_Train_v2',
        name= task_name)

    loss_function_model_GPT2 = nn.CrossEntropyLoss(reduction='none')
    optimizer_model_GPT2 = optim.AdamW(model_GPT2.parameters(), lr=args.learning_rate)

    n_val=int(2e3)
    seed_task= 2 ** 32 - 1


    print('***generate Rayleigh channel data!')
    x_val, y_val, _, _ = generate_signals(n_val, n_tasks, args=args, seed_task=seed_task, seed_example=0)


    x_val, y_val = complex_to_vec(x_val),complex_to_vec(y_val)
    bsize, length, dim = x_val.shape
    encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)

    reshaped_x_val = np.array([str(x) for x in x_val.reshape(-1, dim)]).reshape(-1, 1)
    encoder.fit(reshaped_x_val)

    x_val_encoded = encoder.transform(reshaped_x_val)
    num_classes = x_val_encoded.shape[1]
    x_val_encoded = x_val_encoded.reshape(bsize, length, num_classes)


    n_it_per_epoch=10
    LOG=[]
    log_every=10
    best_val=100
    best_it=0

    SAVE_MODEL = True

    for jj in range(args.epochs):
        seed_task =1
        x_tr,y_tr, _, _ = generate_signals(int(args.batch_size * n_it_per_epoch), n_tasks, args=args
                                             , seed_task=seed_task, seed_example=jj)
        x_tr,  y_tr = complex_to_vec(x_tr),complex_to_vec(y_tr)
        bsize, length, dim = x_tr.shape
        reshaped_x_tr = np.array([str(x) for x in x_tr.reshape(-1, dim)]).reshape(-1, 1)
        x_tr_encoded = encoder.transform(reshaped_x_tr)
        num_classes = x_tr_encoded.shape[1]
        x_tr_encoded = x_tr_encoded.reshape(-1, length, num_classes)

        running_loss=0
        for ii in range(int(n_it_per_epoch)):
            batch_id=np.arange(ii*args.batch_size,(ii+1)*args.batch_size)
            if args.DFE_TRAIN:
                if jj == 0 and ii == 0:
                    print('*** Start DFE Train: ' + task_name)
                if jj< args.DFE_epoch:
                    loss, output = train_step(model_GPT2, ys_batch=torch.Tensor(y_tr[batch_id, :, :]).to(device),
                                              xs_batch=torch.Tensor(x_tr_encoded[batch_id, :, :]).to(device),
                                              xs_real=torch.tensor(x_tr_encoded[batch_id, :], dtype=torch.long).to(
                                                  device),
                                              optimizer=optimizer_model_GPT2, loss_func=loss_function_model_GPT2)
                else:
                    loss, output = sequence_train_step(args,model_GPT2, ys_batch=torch.Tensor(y_tr[batch_id, :,:]).to(device),
                                                   xs_batch=torch.Tensor(x_tr_encoded[batch_id, :, :]).to(device),
                                                   xs_real=torch.tensor(x_tr_encoded[batch_id, :], dtype=torch.long).to(
                                                       device), optimizer=optimizer_model_GPT2,
                                                   loss_func=loss_function_model_GPT2,
                                                   train_pilot_len=args.train_pilot_len,
                                                   label_encoder=encoder)
            else:
                if jj == 0 and ii == 0:
                    print('*** Start ICL Train: ' + task_name)
                loss, output = train_step(model_GPT2, ys_batch=torch.Tensor(y_tr[batch_id, :, :]).to(device),
                                          xs_batch=torch.Tensor(x_tr_encoded[batch_id, :, :]).to(device),
                                          xs_real=torch.tensor(x_tr_encoded[batch_id, :], dtype=torch.long).to(device),
                                          optimizer=optimizer_model_GPT2, loss_func=loss_function_model_GPT2)
            running_loss = running_loss + loss/n_it_per_epoch
        wandb.log({'Train: Cross-Entropy Loss': running_loss})


        if SAVE_MODEL and jj%200==0:
            time_id = time.strftime("%m%d%H%M%S", time.localtime())
            torch.save(model_GPT2, '../models_SISO_v2/'+task_name+'_Epoch'+str(jj)+'_'+time_id+'.pth')
    wandb.finish()
    return


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


args = parameter_reading()

args.num_head= 8
args.num_layer = 8
args.embedding_dim = 128
args.embedding_dim_single = 128


args.prompt_seq_length = 31
args.batch_size = 512
args.epochs = 8000


args.num_ant = 1
args.data_dim = 2


args.velocity = 1

n_tasks = 32768
args.bits = 4

args.Clarke_Model = False

args.DFE_TRAIN = True
args.modulation = '16QAM'
args.train_pilot_len = 2
SNR = 10
args.SNR_dB_min = 10
args.SNR_dB_max = 20
args.modu_num = count_modulation_symbols(args)
args.loss_weight = 0.7
args.ISI_weight = 0.5

args.DFE_epoch = 2500


task_name = ((str(args.DFE_epoch) + 'DFE' + str(
            args.loss_weight) + '_' + args.modulation +
                      '_SNR[' + str(args.SNR_dB_min) + ',' + str(args.SNR_dB_max) + ']_SISO_P' + str(
                    args.train_pilot_len) + '_Seq' + str(args.prompt_seq_length) +
                      'Task' + str(n_tasks) + '_Layer' + str(args.num_layer) + 'Embed' + str(
                    args.embedding_dim) + 'Head' + str(args.num_head)))

model = build_model(embedding_dim=args.embedding_dim, n_positions=args.prompt_seq_length, num_heads=args.num_head,
                        num_layers=args.num_layer, data_dim=args.data_dim, num_classes=args.modu_num).to(device)


print(args)

for name, param in model.named_parameters():
    print(f"{name}: {param.numel()}")


total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"***Total number of parameters in the model: {total_trainable_params}")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.numel()}")

trainNetwork(model.to(device), args, n_tasks=n_tasks, task_name=task_name)  

print("***Training is done:"+task_name) 


