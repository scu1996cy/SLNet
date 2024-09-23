import os, utils, losses
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models.SLNet import SLNet
import time

def main():
    test_dir = '/home/SLNet/OASIS/OASIS_L2R_2021_task03/'
    model_idx = -1
    weights = [1, 1]
    model_folder = 'SLNet_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    img_size= (160,192,224)

    model = SLNet()
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()

    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    test_set = datasets.OASISBrainTestDataset(test_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    times = []
    dscs = []
    hds = []
    asds = []
    eval_dsc = utils.AverageMeter()
    eval_hd = utils.AverageMeter()
    eval_asd = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            start = time.time()
            output, flow = model(x, y)
            times.append(time.time() - start)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            dsc, hd, asd = utils.metric_val_VOI(def_out.long(), y_seg.long())
            dscs.append(dsc)
            hds.append(hd)
            asds.append(asd)
            eval_dsc.update(dsc.item(), x.size(0))
            eval_hd.update(hd.item(), x.size(0))
            eval_asd.update(asd.item(), x.size(0))
            print('dsc', eval_dsc.avg)
            print('hd', eval_hd.avg)
            print('asd', eval_asd.avg)
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

    print(np.mean(times[1:]))
    print('dsc{:.4f}dsc_std{:.4f}'.format(eval_dsc.avg,eval_dsc.std))
    print('hd{:.4f}hd_std{:.4f}'.format(eval_hd.avg,eval_hd.std))
    print('asd{:.4f}asd_std{:.4f}'.format(eval_asd.avg,eval_asd.std))
    print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    torch.manual_seed(0)
    main()