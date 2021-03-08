import argparse
import locale
import logging
import os
import random
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable

from functions import *
from models import *
from models.MobileNetV2_quant import mobilenet_v2
from utils import *
import math

parser = argparse.ArgumentParser(description='PyTorch - Learning Quantization')
parser.add_argument('--model', default='mobilenetv2', help='select model')
parser.add_argument('--dir', default='/data', help='data root')
parser.add_argument('--dataset', default='imagenet', help='select dataset')

parser.add_argument('--batchsize', default=64, type=int, help='set batch size')
parser.add_argument("--lr", default=0.005, type=float)
parser.add_argument("--lr_quant", default=0.005, type=float)
parser.add_argument("--lr_w_theta", default=0.005, type=float)
parser.add_argument("--lr_a_theta", default=0.005, type=float)
parser.add_argument("--lr_bn", default=0.005, type=float)
parser.add_argument('--warmup', default=3, type=int)
parser.add_argument('--epochs', default=15, type=int)

parser.add_argument('--log_interval', default=100, type=int, help='logging interval')
parser.add_argument('--exp', default='test', type=str)
parser.add_argument('--seed', default=7, type=int, help='random seed')
parser.add_argument("--quant_op", required=True)

#parser.add_argument('--comp_ratio', default=1, type=float, help='set target compression ratio of Bitops loss')
parser.add_argument('--w_target_bit', default=4, type=float, help='set target weight bitwidth')
parser.add_argument('--a_target_bit', default=4, type=float, help='set target activation bitwidth')
parser.add_argument('--w_bit', default=[32], type=int, nargs='+', help='set weight bits')
parser.add_argument('--a_bit', default=[32], type=int, nargs='+', help='set activation bits')
#parser.add_argument('--scaling', default=1e-6, type=float, help='set FLOPs loss scaling factor')


parser.add_argument('--eval', action='store_true', help='evaluation mode')
parser.add_argument('--lb_off', '-lboff', action='store_true', help='learn bitwidth (dnas approach)')
parser.add_argument('--cooltime', default=0, type=int, help='seconds for processor cooling (for sv8 and sv9')
parser.add_argument('--w_ep', default=1, type=int, help='')
parser.add_argument('--t_ep', default=1, type=int, help='')
parser.add_argument('--alternate', action="store_true")
parser.add_argument('--retrain_path', default='', type=str, help='logged weight path to retrain')
parser.add_argument('--retrain_type', default=1, type=int, help='1: init weight using searched net\n'\
                                                            '2: init weight using pretrained\n'\
                                                            '3: init weight using searched net, reinit quantizer\n'\
                                                            '4: init weight using searched net, fixed bitwidth')
parser.add_argument('--fasttest', action='store_true')
parser.add_argument('--grad_scale', action='store_true')
#parser.add_argument('--lr_q_scale', default=1, type=float, help='lr_quant = args.lr * args.lr_q_scale')
parser.add_argument('--tau_init', default=1, type=float, help='softmax tau (initial)')
parser.add_argument('--tau_target', default=1, type=float, help='softmax tau (final)')
parser.add_argument('--tau_type', default='linear', type=str, help='softmax tau annealing type [linear, exponential, cosine, exp_cyclic]')
parser.add_argument('--alpha_init', default=1e-10, type=float, help='bitops scaling factor (initial)')
parser.add_argument('--alpha_target', default=1e-10, type=float, help='bitops scaling factor (final)')
parser.add_argument('--alpha_type', default='linear', type=str, help='bitops scaling factor annealing type [linear, exponential, cosine]')
parser.add_argument('--cycle_epoch', default=5, type=int, help='Training Cycle size until changing target compression')
parser.add_argument('--temp_step', default=1e-2, type=float, help='Exp factor of the temperature decay')
parser.add_argument('--n_gumbel', default=25, type=int, help='Number until rounding of single temperature step')



args = parser.parse_args()

#------------ For debuging and testing ------------

args.stop_step = 200
#args.lr_a_theta = 0.01
#args.lr_w_theta = 0.01
#args.batchsize = 8

#--------------------------------------------------


if args.exp == 'test':
    args.save = f'logs/{args.dataset}/{args.exp}-{time.strftime("%y%m%d-%H%M%S")}'
else:
    args.save = f'logs/{args.dataset}/{args.exp}' #-{time.strftime("%y%m%d-%H%M%S")}'

#args.bitops_scaledown=1e-09
args.workers = 8
args.momentum = 0.9   # momentum value
args.decay = 5e-4 # weight decay value
args.lb_mode = False
args.comp_ratio = args.w_target_bit / 32. * args.a_target_bit / 32.


if (len(args.w_bit) > 1 or len(args.a_bit) > 1) and not args.lb_off:
    args.lb_mode = True
    print("## Learning bitwidth selection")

create_exp_dir(args.save)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Argument logging ####################
string_to_log = '==> parsed arguments.. \n'
for key in vars(args):
    string_to_log += f'  {key} : {getattr(args, key)}\n'
logging.info(string_to_log)


if len(args.w_bit)==1:
    print("## Fixed bitwidth for weight")

if len(args.a_bit)==1:
    print("## Fixed bitwidth for activation")

if args.lb_mode:
    logging.info("## Learning layer-wise bitwidth.")


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

best_acc = 0
last_epoch = 0
end_epoch = args.epochs
tau = args.tau_init
alpha = args.alpha_init

# Dataloader
print('==> Preparing Data..')
train_loader, val_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)


print('==> Building Model..')
# QuantOps
if args.grad_scale:
    from functions.duq_gscale import *
    print("==> duq with grad_scale is selected..")
elif args.quant_op == "duq":
    from functions.duq import *
    print("==> differentiable and unified quantization method is selected..")
elif args.quant_op == "hwgq":
    from functions.hwgq import *
    print("==> Non-learnineg based HWGQ quantizer is selected..")
elif args.quant_op == "qil":
    torch.autograd.set_detect_anomaly(True)
    from functions.qil import * 
    print("==> quantization interval learning method is selected..")
elif args.quant_op == "lsq":
    from functions.lsq import *
    print("==> learning step size method is selected..")
elif args.quant_op == 'duq_wo_scale':
    from functions.duq_wo_scale import *
    print("==> differentiable and unified quantization without scale.. ")
elif args.quant_op == 'duq_w_offset':
    from functions.duq_w_offset import *
    print("==> differentiable and unified quantization with offset.. ")
elif args.quant_op == 'duq_init_change':
    from functions.duq_init_change import *
else:
    raise NotImplementedError


# calculate bitops for full precision
def get_bitops_fullp(idx=-1):
    model_ = mobilenet_v2(QuantOps)
    model_ = model_.to(device)
    if args.dataset in ["cifar100", "cifar10"]:
        input = torch.randn([1,3,32,32]).cuda()
    else:
        input = torch.randn([1,3,224,224]).cuda()
    model_.train()
    QuantOps.initialize(model_, train_loader, 32, weight=True)
    QuantOps.initialize(model_, train_loader, 32, act=True)
    model_.eval()    
    return get_bitops(model_, idx)


# calculate bitops (theta-weighted)
def get_bitops(model, idx=-1):
    a_bit_list = [32]
    w_bit_list = []
    computation_list = []
    
    for module in model.modules():
        if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)):
            if isinstance(module.bits, int) :
                a_bit_list.append(module.bits)
            else:
                softmask = F.softmax(module.theta/tau, dim=0)
                a_bit_list.append((softmask * module.bits).sum())

        elif isinstance(module, (Q_Conv2d, Q_Linear)):
            if isinstance(module.bits, int) :
                w_bit_list.append(module.bits)
            else:
                softmask = F.softmax(module.theta/tau, dim=0)
                w_bit_list.append((softmask * module.bits).sum())
                
            computation_list.append(module.computation)
    cost = Tensor([0]).cuda()
    if idx < 0:
        for i in range(len(a_bit_list)):
            try:
                cost += a_bit_list[i] * w_bit_list[i] * computation_list[i]
            except Exception as e:
                print(computation_list[i].device())
                print(w_bit_list[i].device())
                print(a_bit_list[i].device())
                exit()
            

    else:
        cost = a_bit_list[idx] * w_bit_list[idx] * computation_list[idx]
    #if idx < 0:
    #    cost = (Tensor(a_bit_list) * Tensor(w_bit_list) * Tensor(computation_list)).sum(dim=0, keepdim=True)
    
    return cost


"""
def get_bitops(model_, device):
    if args.dataset in ["cifar100", "cifar10"]:
        input = torch.randn([1,3,32,32]).to(device)
    else:
        input = torch.randn([1,3,224,224]).to(device)
    model_.eval()
    _, bitops = model_(input)

    return bitops
"""

print("==> Calculate bitops..")
if args.fasttest:
    bitops_total = 307992854528
    bitops_first_layer = 11098128384
if not args.fasttest:
    bitops_total = get_bitops_fullp().item()
    bitops_first_layer = get_bitops_fullp(0).item()
bitops_target = ((bitops_total - bitops_first_layer) * (args.w_target_bit/32.) * (args.a_target_bit/32.) +\
                 (bitops_first_layer * (args.w_target_bit/32.)))

logging.info(f'bitops_total : {int(bitops_total):d}')
logging.info(f'bitops_target: {int(bitops_target):d}')
#logging.info(f'bitops_wrong : {int(bitops_total * (args.w_target_bit/32.) * (args.a_target_bit/32.)):d}')
if type(bitops_target) != float:
    bitops_target = float(bitops_target)

# model
if args.model == "mobilenetv2":
    model = mobilenet_v2(QuantOps)
    if not os.path.isfile("./checkpoint/mobilenet_v2-b0353104.pth"):
        os.system("wget -P ./checkpoint https://download.pytorch.org/models/mobilenet_v2-b0353104.pth")
    model.load_state_dict(torch.load("./checkpoint/mobilenet_v2-b0353104.pth"), False)
    print("pretrained weight is loaded.")
else:
    raise NotImplementedError
model = model.to(device)


# optimizer -> for further coding (got from PROFIT)
def get_optimizer(params, train_weight, train_quant, train_bnbias, train_w_theta, train_a_theta):
    #global lr_quant
    (weight, quant, bnbias, theta_w, theta_a, skip) = params
    optimizer = optim.SGD([
        {'params': weight, 'weight_decay': args.decay, 'lr': args.lr  if train_weight else 0},
        {'params': quant, 'weight_decay': 0., 'lr': args.lr_quant if train_quant else 0},
        {'params': bnbias, 'weight_decay': 0., 'lr': args.lr_bn if train_bnbias else 0},
        {'params': theta_w, 'weight_decay': 0., 'lr': args.lr_w_theta if train_w_theta else 0},
        {'params': theta_a, 'weight_decay': 0., 'lr': args.lr_a_theta if train_a_theta else 0},
        {'params': skip, 'weight_decay': 0, 'lr': 0},
    ], momentum=args.momentum, nesterov=True)
    return optimizer


def categorize_param(model):
    weight = []
    quant = []
    bnbias = []
    theta_w = []
    theta_a = []
    skip = []
    for name, module in model.named_modules():
        if isinstance(module, (QuantOps.Conv2d, QuantOps.Linear)):
            theta_w.append(module.theta)
        elif isinstance(module, (QuantOps.ReLU, QuantOps.ReLU6, QuantOps.Sym)):
            theta_a.append(module.theta)
        
    for name, param in model.named_parameters():
        if name.endswith(".a") or name.endswith(".b") \
            or name.endswith(".c") or name.endswith(".d"):
            quant.append(param)
        elif len(param.shape) == 1 and ((name.endswith('weight') or name.endswith(".bias"))):
            bnbias.append(param)
        elif name.endswith(".theta"):
            pass
            #theta.append(param)
        else:
            weight.append(param)
    return (weight, quant, bnbias, theta_w, theta_a, skip,)



# Bitwidth Initilization 
with torch.no_grad():
    # Retraining based on searched bitwidths
    if args.retrain_path:
        checkpoint = torch.load(args.retrain_path)
        if "model" in checkpoint.keys():
            checkpoint = checkpoint["model"]
        
        for key in checkpoint:
            #print(key)
            if 'conv.0.0.bits' in key:
                num_w_bits = len(checkpoint[key])
                break
        for key in checkpoint:
            if 'conv.-0.bits' in key:
                num_a_bits = len(checkpoint[key])
                break
        dummy_w_bits = [i for i in range(3, 3+num_w_bits)]
        dummy_a_bits = [i for i in range(3, 3+num_a_bits)]

        if args.retrain_type == 1:
            logging.info('[Retraining type 1] sample weight and quantizer parameter from bitsearch result')
            QuantOps.initialize(model, train_loader, dummy_w_bits, weight=True)
            QuantOps.initialize(model, train_loader, dummy_a_bits, act=True)
            
            print('==> load searched result..')
            model.load_state_dict(checkpoint)
            print('==> sample search result..')
            sample_search_result(model)
        
        elif args.retrain_type == 2:
            logging.info('[Retraining type 2] load weight from pretrained model, and just change bitwidths')
            print('=> Initialize for main model..')
            QuantOps.initialize(model, train_loader, dummy_w_bits[0], weight=True)
            QuantOps.initialize(model, train_loader, dummy_a_bits[0], act=True)
            
            model2 = copy.deepcopy(model) 
            print('=> Initialize for auxilary model..')
            QuantOps.initialize(model2, train_loader, dummy_w_bits, weight=True)
            QuantOps.initialize(model2, train_loader, dummy_a_bits, act=True)
            model2.load_state_dict(checkpoint)
            
            print('==> Sample search result..')
            sample_search_result(model2)
            print('==> Transfer searched result to main model..')
            transfer_bitwidth(model2, model)
            del(model2)

        elif args.retrain_type == 3:
            logging.info('[Retraining type 3] sample weight from bitsearch result, and reinitialize a and c')
            print('=> Initialize for main model..')
            QuantOps.initialize(model, train_loader, dummy_w_bits, weight=True)
            QuantOps.initialize(model, train_loader, dummy_a_bits, act=True)
            model.load_state_dict(checkpoint)

            print('==> Sample search result..')
            sample_search_result(model)
            model = model.to(device)
            QuantOps.initialize_quantizer(model, train_loader)

        elif args.retrain_type == 4:
            logging.info('[Retraining type 4] initialize weight using bitsearch result, training fixed bitwidth')
            QuantOps.initialize(model, train_loader, dummy_w_bits, weight=True)
            QuantOps.initialize(model, train_loader, dummy_a_bits, act=True)
            model.load_state_dict(checkpoint, strict=False)
            QuantOps.initialize(model, train_loader, args.w_bit, weight=True)
            QuantOps.initialize(model, train_loader, args.a_bit, act=True)


        # ---- Variable interval (end) ---------------------------
        _, _, str_sel, _ = extract_bitwidth(model, weight_or_act="weight")
        logging.info(str_sel)
        _, _, str_sel, _ = extract_bitwidth(model, weight_or_act="act")
        logging.info(str_sel)

        model = model.to(device)
        logging.info(f"## sampled model bitops: {int(get_bitops(model).item())}")

    
    # Bitwidth searching or uniform QAT
    else:
        print('==> weight bitwidth is set up..')
        QuantOps.initialize(model, train_loader, args.w_bit, weight=True)
        print('==> activation bitwidth is set up..')
        QuantOps.initialize(model, train_loader, args.a_bit, act=True)


model = model.to(device)
if torch.cuda.device_count() > 1:
    print(f'==> DataParallel: device count = {torch.cuda.device_count()}')
    model = torch.nn.DataParallel(model) #, device_ids=range(torch.cuda.device_count()))



# optimizer & scheduler

params = categorize_param(model)
optimizer = get_optimizer(params, True, True, True, True, True)
current_lr = -1

scheduler = CosineWithWarmup(optimizer, 
        warmup_len=args.warmup, warmup_start_multiplier=0.1,
        max_epochs=args.epochs, eta_min=1e-3)

criterion = nn.CrossEntropyLoss()

w_module_nums = [4, 32, 36, 39, 123, 127, 130] if len(args.w_bit) > 1 else []
a_module_nums = [6, 30, 34, 38, 121, 125, 129] if len(args.a_bit) > 1 else []
temp_func = get_exp_cyclic_annealing_tau(args.cycle_epoch * len(train_loader),
                                         args.temp_step,
                                         np.round(len(train_loader) / args.n_gumbel))

_, _, _, str_prob = extract_bitwidth(model, weight_or_act=0)
print(str_prob)
_, _, _, str_prob = extract_bitwidth(model, weight_or_act=1)
print(str_prob)

# Training
def train(epoch, phase=None):
    
    global tau
    logging.info(f'[{phase}] train:')
    for i in range(len(optimizer.param_groups)):
        logging.info(f'[epoch {epoch}] optimizer, lr{i} = {optimizer.param_groups[i]["lr"]:.6f}')
    model.train()
    eval_acc_loss = AverageMeter()
    eval_bitops_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    #current_lr = optimizer.param_groups[0]['lr']
    
    end = t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #if batch_idx > args.stop_step:
        #    break
        tau_p = tau
        tau = temp_func((epoch - 1) * len(train_loader) + batch_idx)
        if tau_p != tau:
            set_tau(QuantOps, model, tau)
        '''
        if args.lb_mode and args.alternate:
            if batch_idx % 1000 == 0: # learning weight
                optimizer.param_groups[0]['lr'] = current_lr
                optimizer.param_groups[1]['lr'] = current_lr if args.grad_scale else current_lr * 1e-2
                optimizer.param_groups[3]['lr'] = 0   # 0:weight  1:quant  2:bnbias  3:theta
                # TODO: consider quantizer parameter lr
                
            elif batch_idx % 1000 == 800: # learning theta
                optimizer.param_groups[0]['lr'] = 0
                optimizer.param_groups[1]['lr'] = 0
                optimizer.param_groups[3]['lr'] = current_lr  # 0:weight  1:quant  2:bnbias  3:theta
                # TODO: consider quantizer parameter lr
        '''

        inputs, targets = inputs.to(device), targets.to(device)
        data_time = time.time()
        outputs = model(inputs)
        bitops = get_bitops(model)


        loss = criterion(outputs, targets)
        eval_acc_loss.update(loss.item(), inputs.size(0))
        
        if args.lb_mode and optimizer.param_groups[3]['lr'] != 0 :#(epoch-1) % (args.w_ep + args.t_ep) >= args.w_ep:
            #if not isinstance(bitops, (float, int)):
            #    bitops = bitops.mean()
            loss_bitops = F.relu((bitops - bitops_target)/bitops_target * alpha).reshape(torch.Size([]))
            loss += loss_bitops 
            #loss = loss_bitops
            eval_bitops_loss.update(loss_bitops.item(), inputs.size(0))

        acc1, acc5 = accuracy(outputs.data, targets.data, top_k=(1,5))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_time = time.time()
        if (batch_idx) % args.log_interval == 0:
            tau = temp_func(batch_idx)
            logging.info('Train Epoch: %4d Process: %5d/%5d  ' + \
                    'L_acc: %.3f | L_bitops: %.3f | top1.avg: %.3f%% | top5.avg: %.3f%% | ' +  \
                    'Data Time: %.3f s | Model Time: %.3f s',   # \t Memory %.03fMB',
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                eval_acc_loss.avg, eval_bitops_loss.avg if optimizer.param_groups[3]['lr'] !=0 else 0, 
                top1.avg, top5.avg,
                data_time - end, model_time - data_time)
            if args.cooltime and epoch != end_epoch:
                print(f'> [sleep] {args.cooltime}s for cooling GPUs.. ', end='', flush=True)
                time.sleep(args.cooltime)
                print('done.')
            
            ###  log theta of several selected modules
            i = 1
            fw_list = []
            fa_list = []
            for num in w_module_nums:
                fw_list.append(open(os.path.join(args.save, f'w_{num}_theta.txt'), mode='a'))
            for num in a_module_nums:
                fa_list.append(open(os.path.join(args.save, f'a_{num}_theta.txt'), mode='a'))
            for m in model.modules():
                if i in w_module_nums:
                    prob = F.softmax(m.theta / tau, dim=0)
                    prob = [f'{i:.5f}' for i in prob.cpu().tolist()]
                    prob = ", ".join(prob) + "\n"
                    fw_list[w_module_nums.index(i)].write(prob)
                    #print(f"[DEBUG] {w_module_nums.index(i)}-th weight module is wrote")

                elif i in a_module_nums:
                    prob = F.softmax(m.theta / tau, dim=0)
                    prob = [f'{i:.5f}' for i in prob.cpu().tolist()]
                    prob = ", ".join(prob) + "\n"
                    fa_list[a_module_nums.index(i)].write(prob)
                    #print(f"[DEBUG] {a_module_nums.index(i)}-th activation module is wrote")
                i += 1
            for f_ in fw_list + fa_list:
                f_.close()
        end = time.time()

    #optimizer.param_groups[0]['lr'] = current_lr
    #optimizer.param_groups[3]['lr'] = current_lr 
    if args.lb_mode:
        _, _, str_select, str_prob = extract_bitwidth(model, weight_or_act="weight", tau=tau)
        logging.info(f'Epoch {epoch}, weight bitwidth selection \n' + \
                      str_select + '\n'+ str_prob)
        
        _, _, str_select, str_prob = extract_bitwidth(model, weight_or_act="act", tau=tau)
        logging.info(f'Epoch {epoch}, activation bitwidth selection probability: \n' + \
                      str_select + '\n'+ str_prob)
        
    t1 = time.time()
    logging.info(f'epoch time: {t1-t0:.3f} s')


def eval(epoch):
    print('eval:')
    global best_acc
    model.eval()
    eval_acc_loss = AverageMeter()
    eval_bitops_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            #if batch_idx > args.stop_step:
            #    break
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            bitops = get_bitops(model)
            eval_acc_loss.update(loss.item(), inputs.size(0))

            if args.lb_mode:
                #if not isinstance(bitops, (float, int)):
                #    bitops = bitops.mean()
                loss_bitops = F.relu((bitops - bitops_target)/bitops_target * alpha).reshape(torch.Size([]))
                loss += loss_bitops 
                eval_bitops_loss.update(loss_bitops.item(), inputs.size(0))
                if (batch_idx) % (args.log_interval*5) == 0:
                    logging.info(f'bitops_target: {bitops_target}')
                    logging.info(f'evalaution time bitops: {bitops}')

            acc1, acc5 = accuracy(outputs.data, targets.data, top_k=(1,5))
            
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            if (batch_idx) % args.log_interval == 0:
                logging.info('Train Epoch: %4d Process: %5d/%5d  ' + \
                        'L_acc: %.3f | L_bitops: %.3f | top1.avg: %.3f%% | top5.avg: %.3f%% | ',
                    epoch, batch_idx * len(inputs), len(val_loader.dataset),
                    eval_acc_loss.avg, eval_bitops_loss.avg, top1.avg.item(), top5.avg.item())
                if args.cooltime and epoch != end_epoch:
                    print(f'> [sleep] {args.cooltime}s for cooling GPUs.. ', end='')
                    time.sleep(args.cooltime)
                    print('done.')

        logging.info('L_acc: %.4f | L_bitops: %.3f | top1.avg: %.3f%% | top5.avg: %.3f%%' \
                    % (eval_acc_loss.avg, eval_bitops_loss.avg*1e-9, top1.avg, top5.avg))
        
        # Save checkpoint.        
        is_best = False
        if top1.avg > best_acc:
            is_best = True
            best_acc = top1.avg
        
        create_checkpoint(model, None, optimizer, is_best, None, 
                          top1.avg, best_acc, epoch, args.save, 1, args.exp)
    

if __name__ == '__main__':
    if args.eval:
        eval(0)
    elif args.retrain_path:
        last_epoch, best_acc = resume_checkpoint(model, None, optimizer, scheduler, 
                                        args.save, args.exp)
        for epoch in range(last_epoch+1, end_epoch+1):
            logging.info('Epoch: %d/%d Best_Acc: %.3f' %(epoch, end_epoch, best_acc))
            train(epoch, phase='Retrain')
            eval(epoch)
            scheduler.step()
    else:
        last_epoch, best_acc = resume_checkpoint(model, None, optimizer, scheduler, 
                                        args.save, args.exp)
        
        for epoch in range(last_epoch+1, end_epoch+1):
            alpha, string = get_alpha(args.alpha_init, args.alpha_target, args.epochs, epoch, args.alpha_type)
            logging.info(string)
            if args.tau_type == "exp_cyclic":
                pass
            else:
                tau, string = get_tau(args.tau_init, args.tau_target, args.epochs, epoch, args.tau_type)
                logging.info(string)
                set_tau(QuantOps, model, tau)

            logging.info('Epoch: %d/%d Best_Acc: %.3f' %(epoch, end_epoch, best_acc))
            train(epoch, phase='Search')
            eval(epoch)
            scheduler.step()

            
    logging.info('Best accuracy : {:.3f} %'.format(best_acc))
