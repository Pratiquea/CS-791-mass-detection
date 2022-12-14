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
