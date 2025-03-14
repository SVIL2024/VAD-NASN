import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from data import *
from torch.autograd import Variable
import scipy.stats
from reconstruction_model import *
from utils import *
import random
import argparse
import time
import random
# from mem import *

torch.autograd.set_detect_anomaly(True)

# number_range = [90, 180, 270]
# random_number = random.choice(number_range)


parser = argparse.ArgumentParser(description="EMF")
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--learning_rate_ped2', type=float, default=1e-4, help='initial learning rate')
# parser.add_argument('--learning_rate_ped2', type=float, default=6e-5, help='initial learning rate')
# parser.add_argument('--learning_rate_avenue', default=0.0000001, type=float, help='initial learning_rate')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=0, help='number of workers for the test loader')
# parser.add_argument('--loss_m_weight', help='loss_m_weight', type=float, default=0.0002)

parser.add_argument('--dataset_type', type=str, default='avenue', choices=['ped2', 'avenue', 'shanghai'],
                    help='type of dataset: ped2, avenue, shanghai')
# parser.add_argument('--path', type=str, default='./exp_up13', help='directory of data')
parser.add_argument('--path', type=str, default='exp_siamese', help='directory of data')
parser.add_argument('--path_num', type=int, default=40, help='number of path')
# parser.add_argument('--mem_dim', type=int, default=2000, help='size of mem')
# parser.add_argument('--ano_mem_dim', type=int, default=2000, help='size of mem_ano')
parser.add_argument('--sigma_noise', default='0.9', type=float, help='sigma of noise added to the iamges')

parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='basename of folder to save weights')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                    help='adam or sgd with momentum and cosine annealing lr')
# parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
# parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
# parser.add_argument('--mem_usage', default=[False, False, False, True], type=str)
# parser.add_argument('--skip_ops', default=["none", "concat", "none"], type=str)

parser.add_argument('--pseudo_anomaly_jump', type=float, default=0.01,
                    help='pseudo anomaly jump frame (skip frame) probability. 0 no pseudo anomaly')
parser.add_argument('--jump', nargs='+', type=int, default=[2],
                    help='Jump for pseudo anomaly (hyperparameter s)')  # --jump 2 3

parser.add_argument('--model_dir', type=str, default=None, help='path of model for resume')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch. usually number in filename + 1')

# test
parser.add_argument('--th', type=float, default=0.02, help='threshold for test updating')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--print_score', action='store_true', help='print score')
parser.add_argument('--vid_dir', type=str, default=None, help='save video frames file')
parser.add_argument('--print_time', action='store_true', help='print forward time')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

# Device options
parser.add_argument('--gpu_id', default=[0], type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
use_cuda = torch.cuda.is_available()
np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告

# if args.manualSeed is None:
#     args.manualSeed = random.randint(1, 10000)
# random.seed(args.manualSeed)
# torch.manual_seed(args.manualSeed)
# if use_cuda:
#     torch.cuda.manual_seed_all(args.manualSeed)
#     device = torch.cuda.current_device()
channel_in = 1
if args.dataset_type == 'ped2':
    channel_in = 1
    learning_rate = args.learning_rate_ped2
    train_folder = os.path.join('UCSDped2', 'Train')
    test_folder = os.path.join('UCSDped2', 'Test')
    # args.epochs = 50

if args.dataset_type == 'avenue':
    channel_in = 3
    learning_rate = 1e-4
    train_folder = os.path.join('Avenue', 'Train')
    test_folder = os.path.join('Avenue', 'Test')
    args.epochs = 25

if args.dataset_type == 'shanghai':
    channel_in = 3
    learning_rate = 1e-4
    train_folder = os.path.join('shanghaitech', 'training', 'frames')
    test_folder = os.path.join('shanghaitech', 'testing', 'frames')
    args.epochs = 10

print(f'epochs:{args.epochs}')

exp_dir = args.exp_dir + '_lr' + str(learning_rate) + '_' + 'weight' + '_recon'

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
if torch.cuda.is_available():
    print("GPU可用！")
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
else:
    print("GPU不可用，将使用CPU进行计算。")

# print('exp_dir: ', exp_dir)
# torch.cuda.set_device(args.gpu_id)

# torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance


# train_folder = os.path.join(args.dataset_path, args.dataset_type, 'training')
# train_folder = os.path.join('UCSDped2', 'Train')
# train_folder = os.path.join('Avenue', 'Train')

print(f'train_folder:{train_folder}')

trans_compose = transforms.Compose([transforms.ToTensor()])
log_dir = os.path.join('./', args.path + str(args.path_num), args.dataset_type, exp_dir)
print(f'log_dir:{log_dir}')

# Loading dataset
img_extension = '.tif' if args.dataset_type == 'ped2' else '.jpg'
train_dataset = Reconstruction3DDataLoader(train_folder, trans_compose,
                                           resize_height=args.h, resize_width=args.w, dataset=args.dataset_type,
                                           img_extension=img_extension)
train_dataset_jump = Reconstruction3DDataLoaderJump(train_folder,
                                                    # transforms.Compose([transforms.ToTensor()]),
                                                    trans_compose,
                                                    resize_height=args.h, resize_width=args.w,
                                                    dataset=args.dataset_type,
                                                    jump=args.jump, img_extension=img_extension)

# train_dataset = traindataset(train_folder)


train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, drop_last=True)
train_batch_jump = data.DataLoader(train_dataset_jump, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers, drop_last=True)

print(f'len(train_batch):{len(train_batch)}')

# Report the training process

# while os.path.exists(log_dir):
#     args.path_num = args.path_num + 1
# log_dir = os.path.join('./', args.path + str(args.path_num), args.dataset_type, exp_dir)
# if not os.path.exists(args.path + str(args.path_num)):


if not os.path.exists(log_dir):
    os.makedirs(log_dir)

orig_stdout = sys.stdout

f = open(os.path.join(log_dir, 'log.txt'), 'a')
sys.stdout = f

torch.set_printoptions(profile="full")

loss_func_mse = nn.MSELoss(reduction='none')
# entropy_loss_func = EntropyLossEncap().cuda()
# loss_m_weight = args.loss_m_weight

if args.start_epoch < args.epochs:
    # model = convAE()
    # model = nn.DataParallel(model)
    # model.cuda()
    # new_net = nn.DataParallel(net, device_ids=[0, 1])
    sigma = args.sigma_noise ** 2
    En = Reconstruction3DEncoder(chnum_in=channel_in)
    Den = Reconstruction3DDecoder(chnum_in=channel_in)
    Dep = Reconstruction3DDecoder0(chnum_in=channel_in)
    Dep.load_state_dict(Den.state_dict())
    En = nn.DataParallel(En, device_ids=args.gpu_id)
    En.cuda()
    Den = nn.DataParallel(Den, device_ids=args.gpu_id)
    Den.cuda()
    Dep = nn.DataParallel(Dep, device_ids=args.gpu_id)
    Dep.cuda()
    fea_p = torch.zeros(1, 1).cuda()
    out_p = torch.zeros(1, 1).cuda()
    out_n = torch.zeros(1, 1).cuda()
    out_p1 = torch.zeros(1, 1).cuda()
    out_p2 = torch.zeros(1, 1).cuda()
    out_p3 = torch.zeros(1, 1).cuda()
    kl_fea = torch.zeros(1, 1).cuda()
    kl_label = 0
    out_label = 0

    params_En = list(En.parameters())
    params_Den = list(Den.parameters())
    params = params_En + params_Den

    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.9)

    tic = time.time()
    En.eval()
    Den.eval()
    Dep.eval()

    # model.eval()
    for epoch in range(args.start_epoch, args.epochs):
        pseudolossepoch = 0
        lossepoch = 0
        pseudolosscounter = 0
        losscounter = 0

        for j, (imgs, imgsjump) in enumerate(zip(train_batch, train_batch_jump)):
            net_in = copy.deepcopy(imgs)
            net_in = net_in.cuda()
            # net_in_p = copy.deepcopy(imgs)
            # net_in_p = net_in_p.cuda()
            # imgsjump_app = genAppAnoSmps(imgsjump).cuda()
            imgsjump_gaus = gaussian(imgsjump, 1, 0, sigma).cuda()
            # net_in_p[:] = imgsjump_gaus[:][0]
            jump_pseudo_stat = []
            cls_labels = []

            for b in range(args.batch_size):
                total_pseudo_prob = 0
                rand_number = np.random.rand()
                pseudo_bool = False

                # skip frame pseudo anomaly
                pseudo_anomaly_jump = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_jump
                total_pseudo_prob += args.pseudo_anomaly_jump
                if pseudo_anomaly_jump:
                    # net_in[b] = imgsjump_app[b][0]
                    # net_in[b] = imgsjump[b][0]
                    # net_in[b] = imgsjump_gaus[b][0]
                    net_in[b] = imgsjump_gaus[b]
                    jump_pseudo_stat.append(True)
                    pseudo_bool = True
                else:
                    jump_pseudo_stat.append(False)

                if pseudo_bool:
                    cls_labels.append(0)
                else:
                    cls_labels.append(1)

            ########## TRAIN

            for b in range(args.batch_size):
                if jump_pseudo_stat[b]:
                    fea_p = En(net_in)
                    out_p1, out_p2, out_p3, out = Dep(fea_p)
                    kl_label = 1
                else:
                    fea_n = En(net_in)
                    out_n1, out_n2, out_n3, out = Den(fea_n)

            # outputs = model(net_in)

            cls_labels = torch.Tensor(cls_labels).unsqueeze(1).cuda()






            # loss_feas_mem = -1.0 * torch.abs(fea_p.detach().cuda() - fea_mem['output'].detach().cuda())
            loss_feas = -1.0 * torch.abs(fea_p.detach().cuda() - fea_n.detach().cuda())
            loss_mse = loss_func_mse(out, net_in)
            # loss_sparsity = torch.mean(torch.sum(-fea_mem["att"].detach() * torch.log(fea_mem["att"].detach() + 1e-12), dim=1))
            # loss_sparsity = loss_sparsity.detach()

            # loss_out = -1.0 * torch.abs(out_p.detach().cuda() - out_n.detach().cuda())
            # loss_feas = torch.mean(loss_feas)
            # loss_out = torch.mean(loss_out)
            # loss_feas1 = -1.0 * torch.abs(out_n1.detach().cuda() - out_p1.detach().cuda())
            # loss_feas2 = -1.0 * torch.abs(out_n2.detach().cuda() - out_p2.detach().cuda())
            # loss_feas3 = -1.0 * torch.abs(out_n3.detach().cuda() - out_p3.detach().cuda())

            modified_loss_mse = []
            for b in range(args.batch_size):
                if jump_pseudo_stat[b]:

                    modified_loss_mse.append(torch.mean(-loss_mse[b]))
                    pseudolossepoch += modified_loss_mse[-1].cpu().detach().item()
                    pseudolosscounter += 1
                    # kl_fea = -1.0 * KL_divergence(fea_n.detach().cpu(), fea_p.detach().cpu())
                    # loss_mem = -1.0 * torch.abs(fea_mem['mem'].detach().cuda() - fea_mem_p['mem'].detach().cuda())
                    kl_fea = -1.0 * KL_divergence(fea_n.detach().cpu(), fea_p.detach().cpu())
                    kl_fea = torch.tensor(kl_fea).cuda()
                    #kl_out1 = -1.0 * KL_divergence(out_n1.detach().cpu(), out_p1.detach().cpu())
                    #kl_out1 = torch.tensor(kl_out1).cuda()
                    #kl_out2 = -1.0 * KL_divergence(out_n2.detach().cpu(), out_p2.detach().cpu())
                    #kl_out2 = torch.tensor(kl_out2).cuda()
                    kl_out3 = -1.0 * KL_divergence(out_n3.detach().cpu(), out_p3.detach().cpu())
                    kl_out3 = torch.tensor(kl_out3).cuda()
                    #kl_out4 = -1.0 * KL_divergence(out_n4.detach().cpu(), out_p4.detach().cpu())
                    #kl_out4 = torch.tensor(kl_out4).cuda()
                    kl_label = 1
                    out_p = out[b]

                else:  # no pseudo anomaly
                    # loss_mse_n = loss_func_mse(out_n4, net_in)
                    modified_loss_mse.append(torch.mean(loss_mse[b]))
                    lossepoch += modified_loss_mse[-1].cpu().detach().item()
                    losscounter += 1
                    out_n = out[b]
                    out_label = 1

            # assert len(modified_loss_mse) == loss_mse.size(0)
            stacked_loss_mse = torch.stack(modified_loss_mse)
            if kl_label:
                # kl_out = -1.0 * KL_divergence(out_n.detach().cpu(), out_p.detach().cpu())
                # kl_out = torch.tensor(kl_out)
                # infonce = -1.0 * InfoNCE_loss(fea_n.detach().cpu(), fea_p.detach().cpu())
                # infonce = torch.tensor(infonce).cuda()
                # infonce = infonce.clone().detach().cuda()

                loss = torch.mean(stacked_loss_mse) \
                       + (loss_feas.sum()) * 0.0002 \
                       + kl_fea * 0.00015 \
                       + kl_out3.sum() * 0.00025\
                        #+ kl_out1.sum() * 0.00025\

            #                        + (loss_feas_mem.sum()) * 0.000002 \
            #                        + 0.0002 * loss_sparsity
            #                        + infonce * 0.0002
            #                        + kl_out.sum() * 0.0002 \
            #                        + (loss_mem.sum()) * 0.0002 \

            else:
                loss = torch.mean(stacked_loss_mse) \
                       + (loss_feas.sum()) * 0.0002 \
                    #                        + (loss_mem.sum()) * 0.0002 \
            #                        + (loss_feas_mem.sum()) * 0.000002 \
            #                        + 0.0002 * loss_sparsity

            # if out_label and kl_label:
            #     loss_out = -1.0 * torch.abs(out_n.cpu().detach() - out_p.cpu().detach())
            #     loss_out = loss_out.cuda()
            #     loss = loss + loss_out.sum() * 0.0002

            # + (loss_out.sum()) * 0.0002
            # + (loss_feas3.sum()) * 0.00002
            # + (loss_feas2.sum()) * 0.00002
            # + loss_feas1.sum() * 0.00002

            optimizer.zero_grad()
            loss.sum().backward(retain_graph=True)
            optimizer.step()

        # scheduler.step()

        # Save the model and the memory items
        model_dict = {
            # 'model': model
            'En': En.module.state_dict(),
            # 'mem': mem,
            # 'mem_p': mem_p,
            'Den': Den.module.state_dict(),
            # 'Dep': Dep
        }

        torch.save(model_dict, os.path.join(log_dir, 'model_{:02d}.pth'.format(epoch)))
    print('Training is finished')
    toc = time.time()
    # print('time:' + str(1000 * (toc - tic)) + "ms")
    # print('mean time:' + str(1000 * (toc - tic) / args.epochs) + "ms")
    # print('start:', tic)
    # print('finish:', toc)
    print('time:' + str(1000 * (toc - tic) / 60000 / 60) + "h")
    print('mean time:' + str(1000 * (toc - tic) / 60000 / args.epochs) + "min")
    sys.stdout = orig_stdout
    f.close()

# Test
print('testing..')
loss_func_mse = nn.MSELoss(reduction='none')
labels = np.load('./frame_labels_' + args.dataset_type + '.npy', allow_pickle=True)
img_extension = '.tif' if args.dataset_type == 'ped2' else '.jpg'
test_dataset = Reconstruction3DDataLoader(test_folder, trans_compose,
                                          resize_height=args.h, resize_width=args.w, dataset=args.dataset_type,
                                          img_extension=img_extension, train=False)
test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

# model = convAE()
# model = nn.DataParallel(model)

En = Reconstruction3DEncoder(chnum_in=channel_in)
Den = Reconstruction3DDecoder(chnum_in=channel_in)


print(f'len(test_batch):{len(test_batch)}')
videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*/')))
# videos_list = sorted(glob.glob('test_folder/*'))
# print(videos_list)
for video in videos_list:
    # video_name = video.split('\\')[-2]
    video_name = video.split('\\')[-2]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + img_extension))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

print(f'epochs:{args.epochs}')
print('Evaluation of', args.dataset_type)
# Setting for video anomaly detection
tic = time.time()

for epoch_num in range(args.epochs):
    if epoch_num < 10:
        model_dict = torch.load(log_dir + '\\' + f'model_0{epoch_num}.pth')
    else:
        model_dict = torch.load(log_dir + '\\' + f'model_{epoch_num}.pth')

    # model_weight = model_dict['model']
    # model.load_state_dict(model_dict['model'].state_dict())

    # model.cuda()
    #En = nn.DataParallel(En, device_ids=args.gpu_id)
    #Den = nn.DataParallel(Den, device_ids=args.gpu_id)
    #En.cuda()
    #Den.cuda()
    # En.module.load_state_dict(model_dict['En'].state_dict())
    # Den.module.load_state_dict(model_dict['Den'].state_dict())

    En.load_state_dict(model_dict['En'].state_dict())
    Den.load_state_dict(model_dict['Den'].state_dict())
    En.cuda()
    Den.cuda()

    #En.module.load_state_dict(model_dict['En'])
    #Den.module.load_state_dict(model_dict['Den'])


    for video in sorted(videos_list):
        video_name = video.split('\\')[-2]
        labels_list = np.append(labels_list,
                                labels[0][8 + label_length:videos[video_name]['length'] + label_length - 7])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    label_length = 0
    video_num = 0

    label_length += videos[videos_list[video_num].split('\\')[-2]]['length']

    # model.eval()
    En.eval()
    Den.eval()

    for k, (imgs) in enumerate(test_batch):

        if k == label_length - 15 * (video_num + 1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('\\')[-2]]['length']

        imgs = Variable(imgs).cuda()
        with torch.no_grad():
            # outputs = model(imgs)
            fea = En(imgs)
            # fea_mem = mem(fea.clone())
            out1, out2, out3, out4 = Den(fea)
            loss_mse = loss_func_mse(out4[0, :, 8], imgs[0, :, 8])
            # loss_fea = torch.abs(fea.detach().cuda() - fea_mem['output'].detach().cuda())
        #             loss_fea = 1 - loss_fea_ad
        #             kl_fea = KL_divergence(fea.detach().cpu(), fea_mem['output'].detach().cpu())
        #             kl_fea = torch.tensor(kl_fea).detach()

        loss_pixel = torch.mean(loss_mse)
        mse_imgs = loss_pixel.item()
        # loss_fea = torch.mean(loss_fea)
        # loss_feas = loss_fea.item()

        psnr_list[videos_list[video_num].split('\\')[-2]].append(psnr(mse_imgs))
        # feature_distance_list[videos_list[video_num].split('\\')[-2]].append(loss_feas)

    # Measuring the abnormality score (S) and the AUC
    anomaly_score_total_list = []
    vid_idx = []
    for vi, video in enumerate(sorted(videos_list)):
        video_name = video.split('\\')[-2]
        score = anomaly_score_list(psnr_list[video_name])
        #         anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]),
        #                                               feature_distance_list[video_name], args.alpha)
        anomaly_score_total_list += score
    #         vid_idx += [vi for _ in range(len(score))]
    anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))

    # print('vididx,frame,anomaly_score,anomaly_label')
    # for a in range(len(anomaly_score_total_list)):
    #     print(str(vid_idx[a]), ',', str(a), ',', 1-anomaly_score_total_list[a], ',', labels_list[a])

    print('The result of ', args.dataset_type)
    if epoch_num < 10:
        print(f'model_0{epoch_num}_AUC: ', accuracy * 100, '%')
    else:
        print(f'model_{epoch_num}_AUC: ', accuracy * 100, '%')
    print('----------------------------------------')
toc = time.time()
# print('time:' + str(1000 * (toc - tic)) + "ms")
# print('mean time:' + str(1000 * (toc - tic) / args.epochs) + "ms")
# print('start:', tic)
# print('finish:', toc)
print('time:' + str(1000 * (toc - tic) / 3600000) + "h")
print('mean time:' + str(1000 * (toc - tic) / 60000 / args.epochs) + "min")
print('Testing is finished')
