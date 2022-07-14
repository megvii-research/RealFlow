import argparse
import os
import numpy as np
import torch
import RAFT.core.datasets as datasets
from RAFT.core.utils.frame_utils import writeFlow
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
from torchvision.utils import save_image
from tqdm import tqdm
from utils.tools import FlowReversal
from softmax_splatting import softsplat
from DPT.dpt.models import DPTDepthModel
import imageio
import time
import cv2

@torch.no_grad()
def render_local(flow_net, sample, save_path, alpha, splatting, iters=24):

    #load DPT depth model, using pretrain DPT model
    depth_model_path = "DPT/model/dpt_large-midas-2f21e586.pt"
    DPT = DPTDepthModel(
        path=depth_model_path,
        backbone="vitl16_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    DPT.cuda()
    DPT.eval()

    if not os.path.exists(save_path):
        os.makedirs('{:s}/img1'.format(save_path))
        os.makedirs('{:s}/img2'.format(save_path))
        os.makedirs('{:s}/flow'.format(save_path))


    image1, image2 = sample[0], sample[1]
    image1 = torch.from_numpy(image1).cuda().unsqueeze(0).permute(0,3,1,2).float()
    image2 = torch.from_numpy(image2).cuda().unsqueeze(0).permute(0,3,1,2).float()

    padder = InputPadder(image1.shape, 8)
    image1, image2 = padder.pad(image1, image2)

    # estimate bi-directional flow
    with torch.no_grad():
        _, flow_forward = flow_net(image1, image2, iters=iters, test_mode=True)
        _, flow_back = flow_net(image2, image1, iters=iters, test_mode=True)

    flow_fw = padder.unpad(flow_forward)
    image1 = padder.unpad(image1).contiguous()
    image2 = padder.unpad(image2)
    flow_bw = padder.unpad(flow_back)

    # setting alpha
    linspace = alpha
    flow_fw = flow_fw * linspace
    flow_bw = flow_bw * (1 - linspace)

    # occ check
    with torch.no_grad():
        fw = FlowReversal()
        _, occ = fw.forward(image1, flow_fw)
        occ = torch.clamp(occ, 0, 1)

    # dilated occ mask
    occ = occ.squeeze(0).permute(1,2,0).cpu().numpy()
    occ = (1-occ)*255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(occ, kernel)/255
    occ = 1-torch.from_numpy(dilated).permute(2,0,1).unsqueeze(0).cuda()


    padder = InputPadder(image1.shape, mode='sintel', divisible=32)
    input, input2, flow_fw, flow_bw = padder.pad(image1 / 255, image2 / 255, flow_fw, flow_bw)

    # estimate depth and splatting
    with torch.no_grad():

        # estimate depth and normalize
        tenMetric = DPT(input.cuda())
        tenMetric = (tenMetric - tenMetric.min()) / (tenMetric.max() - tenMetric.min())

        # splatting can choose: softmax, max, summation
        output1 = softsplat.FunctionSoftsplat(tenInput=input, tenFlow=flow_fw,
                                             tenMetric=tenMetric.unsqueeze(0),
                                             strType=splatting)

        tenMetric2 = DPT(input2.cuda())
        tenMetric2 = (tenMetric2 - tenMetric2.min()) / (tenMetric2.max() - tenMetric2.min())
        output2 = softsplat.FunctionSoftsplat(tenInput=input2, tenFlow=flow_bw,
                                              tenMetric=tenMetric2.unsqueeze(0),
                                              strType=splatting)
    # fuse the result
    output = padder.unpad(output1) * occ + (1 - occ) * padder.unpad(output2)
    input = padder.unpad(input)
    flow = padder.unpad(flow_fw).squeeze(0).permute(1, 2, 0).cpu().numpy()
    save_image(input, save_path+'/img1/test_1-1.png')
    save_image(output, save_path+'/img2/RFtest_1-2.png')
    writeFlow(save_path+'/flow/RFflow.flo',flow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # RAFT parameteqqrs
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--save_location', help="save the results in local or oss")
    parser.add_argument('--save_path', help=" local path to save the result")
    parser.add_argument('--iter', help=" kitti 24, sintel 32")
    parser.add_argument('--alpha', default=0.75)
    parser.add_argument('--splatting', help=" max or softmax")
    args = parser.parse_args()


    # load RAFT model
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    img1 = imageio.imread("sample/test_1-1.png").astype('uint8')
    img2 = imageio.imread("sample/test_1-2.png").astype('uint8')
    sample = [img1, img2]

    with torch.no_grad():
        render_local(model, sample, args.save_path, float(args.alpha), str(args.splatting), iters= int(args.iter))
