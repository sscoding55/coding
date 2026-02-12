import os
import argparse
import json
import numpy as np
import cv2
import torch
import pandas as pd
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from medpy.metric.binary import  hd95
from torch.utils.data import DataLoader, Dataset  
from torchvision import transforms               
from PIL import Image     
import os

import math
BatchNorm2d = nn.BatchNorm2d                   
def get_net(net_name,num_classes,ema=False):
    if net_name=='resnet50':
        bkbone = resnet50()
        head = DeepLabHeadV3Plus(in_channels=2048,low_level_channels=256,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)    
    print("build net with encoder {}.".format(net_name))
    return net
def build_global_model(args):
    net_name = args.global_name
    num_classes = args.num_classes
    net = get_net(net_name,num_classes)
    return net
def init_client_nets(args):
    nets_list = {net_i: None for net_i in range(args.num_clients)}
    for net_i in range(args.num_clients):
        net_name = args.clients_model[net_i]
        net = get_net(net_name,args.num_classes)
        nets_list[net_i] = net
      
    return nets_list




class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)
        self.last_conv1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1)
        )
        
        self.confidence_1 = nn.Sequential(
            nn.Conv2d(258, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.ap1 = nn.AdaptiveAvgPool2d(1)
        self.confidence_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.ap2 = nn.AdaptiveAvgPool2d(1)
        self.confidence_3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.confidence_4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
        )
       
        self._init_weight()

    def forward(self, feature,need_fp=False):
        if need_fp:
            output_feature = self.aspp(torch.cat((feature['out'],nn.Dropout2d(0.5)(feature['out']))))
            low_level_feature = self.project(torch.cat((feature['low_level'],nn.Dropout2d(0.5)(feature['low_level']))))
        else:
            output_feature = self.aspp(feature['out'])
            low_level_feature = self.project( feature['low_level'] )
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        output_feature = torch.cat([ low_level_feature, output_feature], dim=1)
        output_feature = self.last_conv1(output_feature)
        con1 = self.confidence_1(torch.cat([output_feature,self.classifier(output_feature)],dim=1))
        con1 = con1+con1*self.ap1(con1)
        con2 = (con1+self.confidence_2(torch.cat([con1,output_feature],dim=1)))
        con2 = con2+con2*self.ap2(con2)
        con3 = (con2+self.confidence_3(torch.cat([con1,con2],dim=1)))
        con4 = self.confidence_4(torch.cat([con3,con2],dim=1))
        
        return {'mask':self.classifier(output_feature),'confidence':con4}
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return 

class SegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(SegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
    def forward(self, x,need_fp=False):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features,need_fp)
        mask = F.interpolate(x['mask'], size=input_shape, mode='bilinear', align_corners=False)
        confidence = F.interpolate(x['confidence'], size=input_shape, mode='bilinear', align_corners=False)
        if need_fp:
            return {'mask':mask.chunk(2),'confidence':confidence,'proto':x['mask'].chunk(2)}
        else:
            return {'mask':mask,'confidence':confidence,'proto':x['mask']}


try: # for torchvision<0.4
    from torchvision.models.utils import load_state_dict_from_url
except: # for torchvision>=0.4
    from torch.hub import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        x = self.layer4(x)


        return {'low_level':x1,'out':x}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model





def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



class SimpleTestDataset(Dataset):
    def __init__(self, args, dataset_name):
        self.root_dir = os.path.join(args.img_path, args.data, dataset_name)
        self.trainsize = args.shape
        
        image_folder = os.path.join(self.root_dir, 'image')
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Could not find folder: {image_folder}")
            
        self.image_list = sorted(os.listdir(image_folder))
        
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        

        image = self.rgb_loader(os.path.join(self.root_dir, 'image', img_name))
        mask = self.binary_loader(os.path.join(self.root_dir, 'mask', img_name))
        
        image, mask = self.resize(image, mask)
        
        seed = 1
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image)
        
        random.seed(seed)
        torch.manual_seed(seed)

        mask = self.gt_transform(mask)
        
        return image, (mask > 0.5).to(torch.float), img_name
   
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
      
            return img.convert('L')
    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        h = self.trainsize 
        w = self.trainsize 
        return img.resize((w, h), Image.Resampling.BILINEAR), gt.resize((w, h), Image.Resampling.NEAREST)

def recursive_sum(lst):
    total = 0
    for item in lst:
        if isinstance(item, list):
            total += recursive_sum(item)  
        else:
            total += item
    return total


def initial_tester(args):
    global_net = build_global_model(args=args)
    net_list = init_client_nets(args)
    
    test_client_dataloaders = []
   
    for ind in range(len(args.datasets)):
       
        client_dataset = SimpleTestDataset(args=args, dataset_name=args.datasets[ind])
        
        client_dataloader = DataLoader(
            client_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
           
        )
        test_client_dataloaders.append(client_dataloader)

    return global_net, net_list, [], test_client_dataloaders


def get_f1(prec,recall,length):
    prec = prec/length
    recall = recall/length
    return 2*prec*recall/(prec+recall)



def dice_score(y_true, y_pred):
    return ((2 * (y_true * y_pred).sum((1,2,3)) + 1e-15) / (y_true.sum((1,2,3)) + y_pred.sum((1,2,3)) + 1e-15)).sum()
def iou_score(y_true, y_pred):
    return (((y_true * y_pred).sum((1,2,3)) + 1e-15) / (y_true.sum((1,2,3)) + y_pred.sum((1,2,3)) -(y_true * y_pred).sum((1,2,3))+ 1e-15)).sum()


@torch.no_grad()
def evaluate_network_fedavg(args,network, dataloaders,key='validation'):
    os.makedirs(os.path.join(args.save_dir,'image'),exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,'mask'),exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,'pred'),exist_ok=True)
    network.eval()
    dice = []
    iou = []
    length = []
    asd = []
    hd95_s = []
    num = []
    over_dice_format = f"result: & "
    over_iou_format  = f" & " 
    over_hd95_format = f" & "
    for i in range(args.num_clients):
        dice.append([])
        iou.append([])
        length.append([])
        asd.append([])
        hd95_s.append([])
        num.append([])
    
    for i in range(len(dataloaders)):
        dataloader = dataloaders[i]
        network.eval()
        for images, labels,img_name in dataloader:
            img_name = img_name[0][:-4]+'_'+str(i)+'_'+args.methods+img_name[0][-4:]
            images = images.cuda()
            labels = labels.cuda()
            pred = network(images)
            mask = pred['mask']
            mask = mask.argmax(1).unsqueeze(1)
            dice[i].append(dice_score(labels, mask).cpu().numpy()*100) 
            iou[i].append(iou_score(labels, mask).cpu().numpy()*100) 
            
            length[i].append(len(labels))
            try:
                hd95_s[i].append(hd95(labels.squeeze(0).squeeze(0).cpu().numpy()>0,mask.squeeze(0).squeeze(0).cpu().numpy()>0))
            except RuntimeError:
                num[i].append(len(labels))
            
            cv2.imwrite(os.path.join(args.save_dir,'image',img_name),((images[0]+1)/2*255)[[2,1,0]].permute(1,2,0).cpu().numpy())
            cv2.imwrite(os.path.join(args.save_dir,'mask',img_name.replace(args.methods,'label')),(labels[0][0]*255).cpu().numpy())
            cv2.imwrite(os.path.join(args.save_dir,'pred',img_name),(mask[0][0]*255).cpu().numpy())
        over_dice_format+= f"& {torch.tensor(dice[i]).mean():.2f} ±  {torch.tensor(dice[i]).std():.2f} "
        over_iou_format  += f"& {torch.tensor(iou[i]).mean():.2f} ± {torch.tensor(iou[i]).std():.2f} "
        over_hd95_format+= f"& {torch.tensor(hd95_s[i]).mean():.2f} ±  {torch.tensor(hd95_s[i]).std():.2f} "
    all_dice = [item for sublist in dice for item in sublist]
    all_iou  = [item for sublist in iou for item in sublist]
    all_hd = [item for sublist in hd95_s for item in sublist]
    over_dice_format+= f"& {torch.tensor(all_dice).mean():.2f} ± {torch.tensor(all_dice).std():.2f} "
    over_iou_format  += f"& {torch.tensor(all_iou).mean():.2f} ± {torch.tensor(all_iou).std():.2f} "
    over_hd95_format+= f"& {torch.tensor(all_hd).mean():.2f} ± {torch.tensor(all_hd).std():.2f} "
    print(over_dice_format+over_iou_format+over_hd95_format)
    return 

def metrics_fedavg(args,global_net,val_client_dataloaders,test_client_dataloaders):
    global_net.load_state_dict(torch.load(os.path.join(args.model_dir,'global_{}_last.pth'.format(args.global_name))))
    global_net = global_net.cuda()           
    print('Evaluate Models')
    print('-'*50)
    print('{} all result:'.format(args.methods))
    evaluate_network_fedavg(args,network=global_net, dataloaders=test_client_dataloaders,key='test')
    print('+'*50)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods',type=str,default='GAFSEG')
    parser.add_argument('--data',type=str,default='test')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--global_name', type=str, default='resnet50',help='[resnet34,resnet50,xception,mobilenetv2,vgg,pvt]')
    parser.add_argument('--clients_model', type=list, default=['resnet50','resnet50','resnet50','resnet50', 'resnet50'],help='[resnet34,xception,mobilenetv2,vgg,pvt]')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--shape', type=tuple, default=384)
    parser.add_argument('--device',type=str,default='0',help="device id")
    parser.add_argument('--img_path', type=str,
                        default='data',
                        help='path to dataset')
    parser.add_argument('--log_dir', type=str,
                        default='log',
                        help='path to log')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    if args.data == "test":
        args.datasets = ['C1', 'C2', 'C3', 'C4', 'C5']
   
    
    
    args.model_dir = f"{args.log_dir}/{args.data}/{args.methods}" # path to checkpoints
    args.save_dir = os.path.join(args.log_dir,'res')
    args.net_name = args.global_name
    global_net,_,val_client_dataloaders,test_client_dataloaders = initial_tester(args)
    metrics_fedavg(args, global_net, val_client_dataloaders, test_client_dataloaders)
    
    
    
    
        
                
