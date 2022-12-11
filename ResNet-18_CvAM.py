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
    def __init__(self, channel_size):
        super(Bi_projection_attention_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        # FC RELU FC
        self.fc = nn.Sequential(nn.Linear(channel_size*4, channel_size, bias=True),
                               nn.ReLU(),
                               nn.Linear(channel_size, channel_size, bias=True))

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
        cat_l = torch.cat((x_l_cc_avg, x_l_cc_max, x_l_mlo_avg, x_l_mlo_max), 1)
        cat_r = torch.cat((x_r_cc_avg, x_r_cc_max, x_r_mlo_avg, x_r_mlo_max), 1)

        flat_l = torch.flatten(cat_l,1)
        flat_r = torch.flatten(cat_r,1)
        fc_l = self.fc(flat_l)
        fc_r = self.fc(flat_r)

        # Sigmoid
        sig_l = self.sigmoid(fc_l)
        sig_r = self.sigmoid(fc_r)
        
        # broadcast value
        output_l = torch.unsqueeze(torch.unsqueeze(sig_l,2), 3)
        output_l = output_l.expand(-1,-1,x_l_cc.shape[2],x_l_cc.shape[3])

        output_r = torch.unsqueeze(torch.unsqueeze(sig_r,2), 3)
        output_r = output_r.expand(-1,-1,x_r_cc.shape[2],x_r_cc.shape[3])
        # print("Bipro l : {}".format(output_l.shape))
        # print("Bipro r: {}".format(output_r.shape))
        return output_l, output_r

class Bi_lateral_attention_module(nn.Module):
    def __init__(self):
        super(Bi_lateral_attention_module, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(nn.Conv2d(4, 4, 3, padding='same'),
                               nn.ReLU(),
                               nn.Conv2d(4, 1, 3, padding='same'))
    def forward(self, x_l_cc, x_r_cc, x_l_mlo, x_r_mlo):
        # Avg Pool
        x_l_cc_avg = torch.mean(x_l_cc,1, True)
        x_r_cc_avg = torch.mean(x_r_cc,1, True)
        x_l_mlo_avg = torch.mean(x_l_mlo,1, True)
        x_r_mlo_avg = torch.mean(x_r_mlo,1, True)
        # print()
        # Max Pool
        x_l_cc_max,_ = torch.max(x_l_cc,1, True)
        x_r_cc_max,_ = torch.max(x_r_cc,1, True)
        x_l_mlo_max,_ = torch.max(x_l_mlo,1, True)
        x_r_mlo_max,_ = torch.max(x_r_mlo,1, True)
        cat_cc = torch.cat([x_r_cc_avg, x_r_cc_max, x_l_cc_avg, x_l_cc_max], dim=1)
        cat_mlo = torch.cat([x_r_mlo_avg, x_r_mlo_max, x_l_mlo_avg, x_l_mlo_max], dim=1)
        
        conv_relu_conv_cc = self.conv(cat_cc)
        conv_relu_conv_mlo = self.conv(cat_mlo)
        sig_cc = self.sigmoid(conv_relu_conv_cc)
        sig_mlo = self.sigmoid(conv_relu_conv_mlo)
        
        output_cc = sig_cc.expand(-1, x_l_cc.shape[1], -1, -1)
        output_mlo = sig_mlo.expand(-1, x_l_mlo.shape[1], -1, -1)

        return sig_cc, sig_mlo

class CvAM(nn.Module):
    def __init__(self, bi_proj_channel_size):
        super(CvAM, self).__init__()
        self.bi_lat = Bi_lateral_attention_module()
        self.bi_pro = Bi_projection_attention_module(bi_proj_channel_size)
    def forward(self, x_l_cc, x_r_cc, x_l_mlo, x_r_mlo):
        # AP_cc, AP_mlo
        bi_lat_cc, bi_lat_mlo = self.bi_lat(x_l_cc, x_r_cc, x_l_mlo, x_r_mlo)
        # AS_l, AS_r
        bi_pro_l, bi_pro_r = self.bi_pro(x_l_cc, x_r_cc, x_l_mlo, x_r_mlo)
        output_x_l_cc = x_l_cc * bi_lat_cc * bi_pro_l
        output_x_r_cc = x_r_cc * bi_lat_cc * bi_pro_r
        output_x_l_mlo = x_l_mlo * bi_lat_mlo * bi_pro_l
        output_x_r_mlo = x_r_mlo * bi_lat_mlo * bi_pro_r
        return [output_x_l_cc, output_x_r_cc, output_x_l_mlo, output_x_r_mlo]

class ResNet18_CVAM(nn.Module):
    def __init__(self):
        super(ResNet18_CVAM, self).__init__()
        # resnet18 = models.resnet18(weights= ResNet18_Weights)
        resnet18 = models.resnet18(pretrained=True)
        a = list(resnet18.children())
        print(list(resnet18.layer1[0].children())[0].weight.shape)
        print(list(resnet18.layer2[0].children())[0].weight.shape)
        print(list(resnet18.layer3[0].children())[0].weight.shape)
        print(list(resnet18.layer4[0].children())[0].weight.shape)
        # print(list(resnet18.layer2))
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

        self.cvam1 = CvAM(list(resnet18.layer1[0].children())[0].weight.shape[0])
        self.cvam2 = CvAM(list(resnet18.layer2[0].children())[0].weight.shape[0])
        self.cvam3 = CvAM(list(resnet18.layer3[0].children())[0].weight.shape[0])

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
        # Flatten layer before this 
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
    x = torch.randn((3, 3, 512, 512))
    # bilinear = Bi_lateral_attention_module()
    # pred1 = bilinear(x, x, x, x)
    # # print("_______")
    # model = Bi_projection_attention_module(x.shape[1])
    # pred = model(x,x,x,x)
    # print("_______")
    # cvam = CvAM(x.shape[1])
    # pred = cvam(x, x, x, x)
    resnet_cvam = ResNet18_CVAM()
    pred = resnet_cvam(x, x, x, x)
    # ytorch_total_params = sum(p.numel() for p in cvam.parameters())
    # print(ytorch_total_params)

    # print(torch.mean(x,1, True).shape)
