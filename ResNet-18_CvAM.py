import torch
from torchvision import models
import torch.nn as nn
from torchsummary import summary
# from torchvision.models.resnet import ResNet18_Weights


resnet18 = models.resnet18(pretrained=True)

# for i in resnet18.children():
#     print("_")
#     print(i)


class Bi_projection_attention_module(nn.Module):
    def __init__(self):
        super(Bi_projection_attention_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_l_cc, x_r_cc, x_l_mlo, x_r_mlo):
        # Avg Pool
        x_l_cc_avg = self.avg_pool(x_l_cc)
        x_r_cc_avg = self.avg_pool(x_r_cc)
        x_l_mlo_avg = self.avg_pool(x_l_mlo)
        x_r_mlo_avg = self.avg_pool(x_r_mlo)
        # Max Pool
        x_l_cc_max = self.max_pool(x_l_cc)
        x_r_cc_max = self.max_pool(x_r_cc)
        x_l_mlo_max = self.max_pool(x_l_mlo)
        x_r_mlo_max = self.max_pool(x_r_mlo)
        # Concat
        cat = torch.cat((x_l_cc_avg, x_r_cc_avg, x_l_cc_max, x_r_cc_max,
                         x_l_mlo_avg, x_r_mlo_avg, x_l_mlo_max, x_r_mlo_max), 1)
        flat = torch.flatten(cat,1)
        # FC RELU FC
        self.fc = nn.Sequential(nn.Linear(flat.shape[-1], flat.shape[-1]//8, bias=True),
                               nn.ReLU(),
                               nn.Linear(flat.shape[-1]//8, flat.shape[-1], bias=True))
        fc = self.fc(flat)

        # Sigmoid
        sig = self.sigmoid(fc)
        # broadcast value
        output = torch.unsqueeze(torch.unsqueeze(sig,1), 2)
        output = output.expand(-1,-1,sig.shape[-1],-1)
        # print("Bipro : {}".format(output.shape))
        return output

class Bi_lateral_attention_module(nn.Module):
    def __init__(self):
        super(Bi_lateral_attention_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(512)
        self.max_pool = nn.AdaptiveMaxPool2d(512)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_l_cc, x_r_cc, x_l_mlo, x_r_mlo):
        # Avg Pool
        x_l_cc_avg = torch.mean(x_l_cc,1, True)
        x_r_cc_avg = torch.mean(x_r_cc,1, True)
        x_l_mlo_avg = torch.mean(x_l_mlo,1, True)
        x_r_mlo_avg = torch.mean(x_r_mlo,1, True)
        # Max Pool
        x_l_cc_max,_ = torch.max(x_l_cc,1, True)
        x_r_cc_max,_ = torch.max(x_r_cc,1, True)
        x_l_mlo_max,_ = torch.max(x_l_mlo,1, True)
        x_r_mlo_max,_ = torch.max(x_r_mlo,1, True)
        cat = torch.cat([x_r_cc_avg, x_r_mlo_avg, x_r_cc_max,x_r_mlo_max,
                         x_l_cc_avg, x_l_mlo_avg, x_l_cc_max,x_l_mlo_max], dim=1)
        [_,in_channel,_,_] = cat.shape
        self.fc = nn.Sequential(nn.Conv2d(in_channel, 4, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(4, 1, 1, bias=False))
        conv_relu_conv = self.fc(cat)
        sig = self.sigmoid(conv_relu_conv)
        # print("Bilat : {}".format(sig.shape))
        return sig

class CvAM(nn.Module):
    def __init__(self):
        super(CvAM, self).__init__()
        self.bi_lat = Bi_lateral_attention_module()
        self.bi_pro = Bi_projection_attention_module()
    def forward(self, x_l_cc, x_r_cc, x_l_mlo, x_r_mlo):
        lat = self.bi_lat(x_l_cc, x_r_cc, x_l_mlo, x_r_mlo)
        pro = self.bi_pro(x_l_cc, x_r_cc, x_l_mlo, x_r_mlo)
        output = [lat * pro * x_l_cc, lat * pro * x_r_cc, lat * pro * x_l_mlo, lat * pro * x_r_mlo]
        return output

class ResNet18_CVAM(nn.Module):
    def __init__(self):
        super(ResNet18_CVAM, self).__init__()
        # resnet18 = models.resnet18(weights= ResNet18_Weights)
        resnet18 = models.resnet18(pretrained=True)
        a = list(resnet18.children())
        self.conv1_1 = a[0]
        self.bn_1 = a[1]
        self.relu_1 = a[2]
        self.pool_1 = a[3]
        self.layer1_1 = a[4]
        self.layer2_1 = a[5]
        self.layer3_1 = a[6]
        self.layer4_1 = a[7]
        self.adaptPool_1 = a[8]
        self.linear_1 = a[9]

        self.conv1_2 = a[0]
        self.bn_2 = a[1]
        self.relu_2 = a[2]
        self.pool_2 = a[3]
        self.layer1_2 = a[4]
        self.layer2_2 = a[5]
        self.layer3_2 = a[6]
        self.layer4_2 = a[7]
        self.adaptPool_2 = a[8]
        self.linear_2 = a[9]

        self.conv1_3 = a[0]
        self.bn_3 = a[1]
        self.relu_3 = a[2]
        self.pool_3 = a[3]
        self.layer1_3 = a[4]
        self.layer2_3 = a[5]
        self.layer3_3 = a[6]
        self.layer4_3 = a[7]
        self.adaptPool_3 = a[8]
        self.linear_3 = a[9]

        self.conv1_4 = a[0]
        self.bn_4 = a[1]
        self.relu_4 = a[2]
        self.pool_4 = a[3]
        self.layer1_4 = a[4]
        self.layer2_4 = a[5]
        self.layer3_4 = a[6]
        self.layer4_4 = a[7]
        self.adaptPool_4 = a[8]
        self.linear_4 = a[9]

        self.cvam1 = CvAM()
        self.cvam2 = CvAM()
        self.cvam3 = CvAM()

    def forward(self, x_l_cc, x_r_cc, x_l_mlo, x_r_mlo):
        x_l_cc_res = self.conv1_1(x_l_cc)
        x_l_cc_res = self.bn_1(x_l_cc_res)
        x_l_cc_res = self.relu_1(x_l_cc_res)
        x_l_cc_res = self.pool_1(x_l_cc_res)
        x_l_cc_res = self.layer1_1(x_l_cc_res)

        x_r_cc_res = self.conv1_1(x_r_cc)
        x_r_cc_res = self.bn_1(x_r_cc_res)
        x_r_cc_res = self.relu_1(x_r_cc_res)
        x_r_cc_res = self.pool_1(x_r_cc_res)
        x_r_cc_res = self.layer1_1(x_r_cc_res)

        x_l_mlo_res = self.conv1_1(x_l_mlo)
        x_l_mlo_res = self.bn_1(x_l_mlo_res)
        x_l_mlo_res = self.relu_1(x_l_mlo_res)
        x_l_mlo_res = self.pool_1(x_l_mlo_res)
        x_l_mlo_res = self.layer1_1(x_l_mlo_res)

        x_r_mlo_res = self.conv1_1(x_r_mlo)
        x_r_mlo_res = self.bn_1(x_r_mlo_res)
        x_r_mlo_res = self.relu_1(x_r_mlo_res)
        x_r_mlo_res = self.pool_1(x_r_mlo_res)
        x_r_mlo_res = self.layer1_1(x_r_mlo_res)

        #CVAM
        [x_l_cc_cvam, x_r_cc_cvam, x_l_mlo_cvam, x_r_mlo_cvam] = self.cvam1(x_l_cc_res, x_r_cc_res, x_l_mlo_res, x_r_mlo_res)

        x_l_cc_res = self.layer2_1(x_l_cc_cvam)
        x_r_cc_res = self.layer2_2(x_r_cc_cvam)
        x_l_mlo_res = self.layer2_3(x_l_mlo_cvam)
        x_r_mlo_res = self.layer2_4(x_r_mlo_cvam)

        # CVAM
        [x_l_cc_cvam, x_r_cc_cvam, x_l_mlo_cvam, x_r_mlo_cvam] = self.cvam2(x_l_cc_res, x_r_cc_res, x_l_mlo_res, x_r_mlo_res)

        x_l_cc_res = self.layer3_1(x_l_cc_cvam)
        x_r_cc_res = self.layer3_2(x_r_cc_cvam)
        x_l_mlo_res = self.layer3_3(x_l_mlo_cvam)
        x_r_mlo_res = self.layer3_4(x_r_mlo_cvam)

        # CVAM
        [x_l_cc_cvam, x_r_cc_cvam, x_l_mlo_cvam, x_r_mlo_cvam] = self.cvam3(x_l_cc_res, x_r_cc_res, x_l_mlo_res, x_r_mlo_res)

        x_l_cc_res = self.layer4_1(x_l_cc_cvam)
        x_l_cc_res = self.adaptPool_1(x_l_cc_res)
        x_l_cc_res = self.linear_1(x_l_cc_res)

        x_r_cc_res = self.layer4_2(x_r_cc_cvam)
        x_r_cc_res = self.adaptPool_1(x_r_cc_res)
        x_r_cc_res = self.linear_1(x_r_cc_res)

        x_l_mlo_res = self.layer4_3(x_l_mlo_cvam)
        x_l_mlo_res = self.adaptPool_3(x_l_mlo_res)
        x_l_mlo_res = self.linear_3(x_l_mlo_res)

        x_r_mlo_res = self.layer4_4(x_r_mlo_cvam)
        x_r_mlo_res = self.adaptPool_3(x_r_mlo_res)
        x_r_mlo_res = self.linear_3(x_r_mlo_res)

        return [x_l_cc_res, x_r_cc_res, x_l_mlo_res, x_r_mlo_res]

if __name__ == "__main__":
    x = torch.randn((3, 64, 512, 512))
    # bilinear = Bi_lateral_attention_module()
    # pred1 = bilinear(x, x, x, x)
    # # print("_______")
    # model = Bi_projection_attention_module()
    # pred = model(x,x,x,x)
    # print("_______")
    # cvam = CvAM()
    # pred = cvam(x, x, x, x)
    # resnet_cvam = ResNet18_CVAM()
    # ytorch_total_params = sum(p.numel() for p in cvam.parameters())
    # print(ytorch_total_params)

    # print(torch.mean(x,1, True).shape)
