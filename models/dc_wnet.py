import torch
import torch.nn as nn
# from .utils import InitWeights_He


class conv(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True))
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        res = x
        x = self.conv(x)
        out = x + res
        out = self.relu(out)
        return x


class feature_fuse(nn.Module):
    def __init__(self, in_c, out_c):
        super(feature_fuse, self).__init__()
        self.conv11 = nn.Conv2d(
            in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.conv33 = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.conv33_di = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv33_di(x)
        out = self.norm(x1+x2+x3)
        return out


class up(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
                               padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=False))

    def forward(self, x):
        x = self.up(x)
        return x


class down(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2,
                      padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.down(x)
        return x


class block_mif(nn.Module):
    def __init__(self, in_c, out_c,  dp=0, is_up=False, is_down=False, fuse=False):
        super(block_mif, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        if fuse == True:
            self.fuse = feature_fuse(in_c, out_c)
        else:
            self.fuse = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)

        self.is_up = is_up
        self.is_down = is_down
        self.conv = conv(out_c, out_c, dp=dp)
        if self.is_up == True:
            self.up = up(out_c, out_c//2)
        if self.is_down == True:
            self.down = down(out_c, out_c*2)

    def forward(self,  x):
        if self.in_c != self.out_c:
            x = self.fuse(x)
        x = self.conv(x)
        if self.is_up == False and self.is_down == False:
            return x
        elif self.is_up == True and self.is_down == False:
            x_up = self.up(x)
            return x, x_up
        elif self.is_up == False and self.is_down == True:
            x_down = self.down(x)
            return x, x_down
        else:
            x_up = self.up(x)
            x_down = self.down(x)
            return x, x_up, x_down


class DC_WNet(nn.Module):
    def __init__(self,  num_classes=1, num_channels=1, feature_scale=2,  dropout=0.2, fuse=True, out_ave=False):
        super(DC_WNet, self).__init__()
        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.block_mif_1_1 = block_mif(
            num_channels, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.new_block_mif_1_1 = block_mif(
            num_channels * 2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block_mif_1_2 = block_mif(
            filters[0], filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block_mif_1_3 = block_mif(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block_mif_1_4 = block_mif(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)
        self.block_mif_1_5 = block_mif(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=True, fuse=fuse)



        self.block_mif_1_6 = block_mif(
            filters[0]*2, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)
        self.block_mif_1_7 = block_mif(
            filters[0]*4, filters[0],  dp=dropout, is_up=False, is_down=False, fuse=fuse)



        self.block_mif_2_1 = block_mif(
            filters[1], filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.new_block_mif_2_1 = block_mif(
            filters[1], filters[1], dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block_mif_2_2 = block_mif(
            filters[1]*2, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block_mif_2_3 = block_mif(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block_mif_2_4 = block_mif(
            filters[1]*3, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block_mif_2_5 = block_mif(
            filters[1]*4, filters[1],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block_mif_3_1 = block_mif(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.new_block_mif_3_1 = block_mif(
            filters[2], filters[2],  dp=dropout, is_up=True, is_down=True, fuse=fuse)
        self.block_mif_3_2 = block_mif(
            filters[2]*2, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block_mif_3_3 = block_mif(
            filters[2]*3, filters[2],  dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.block_mif_4_1 = block_mif(filters[3], filters[3],
                             dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.new_block_mif_4_1 = block_mif(filters[3], filters[3],
                             dp=dropout, is_up=True, is_down=False, fuse=fuse)
        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final4 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.final5 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True)
        self.fuse = nn.Conv2d(
            5, num_classes, kernel_size=1, padding=0, bias=True)


        self.up4_2 = nn.Upsample(scale_factor=1, mode='bilinear')  # 14*14
        # self.up4_2_conv = nn.Conv2d(filters[0]*4, filters[0], 1, padding=1)
        self.up4_2_bn = nn.BatchNorm2d(64)
        self.up4_2_relu = nn.LeakyReLU(0.1, inplace=True)

        self.up4_1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.up4_1_conv = nn.Conv2d(64, 32, 3, padding=1)
        self.up4_1_bn = nn.BatchNorm2d(32)
        self.up4_1_relu = nn.LeakyReLU(0.1, inplace=True)

        self.up3_1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.up3_1_conv = nn.Conv2d(384, 32, 3, padding=1)
        self.up3_1_bn = nn.BatchNorm2d(32)
        self.up3_1_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        #stage 2
        x_1_1, x_down_1_1 = self.block_mif_1_1(x)


        x_plus_4_2 = self.up4_2_relu(self.up4_2_bn(self.up4_2(x_down_1_1)))
        x_plus_4_1 = self.up4_1_relu(self.up4_1_bn(self.up4_1_conv(self.up4_1(x_down_1_1))))


        x_1_2, x_down_1_2 = self.block_mif_1_2(x_1_1)
        x_2_1, x_up_2_1, x_down_2_1 = self.block_mif_2_1(x_down_1_1)
        x_1_3, x_down_1_3 = self.block_mif_1_3(torch.cat([x_1_2, x_up_2_1], dim=1))
        x_2_2, x_up_2_2, x_down_2_2 = self.block_mif_2_2(
            torch.cat([x_down_1_2, x_2_1], dim=1))
        x_3_1, x_up_3_1, x_down_3_1 = self.block_mif_3_1(x_down_2_1)
        x_1_4, x_down_1_4 = self.block_mif_1_4(torch.cat([x_1_3, x_up_2_2], dim=1))
        x_2_3, x_up_2_3, x_down_2_3 = self.block_mif_2_3(
            torch.cat([x_down_1_3, x_2_2, x_up_3_1], dim=1))
        x_3_2, x_up_3_2 = self.block_mif_3_2(torch.cat([x_down_2_2, x_3_1], dim=1))
        old_x_down_3_1, x_up_4_1 = self.block_mif_4_1(x_down_3_1)
        x_1_5, x_down_1_5 = self.block_mif_1_5(torch.cat([x_1_4, x_up_2_3], dim=1))
        x_2_4, x_up_2_4 = self.block_mif_2_4(torch.cat([x_down_1_4, x_2_3, x_up_3_2], dim=1))
        old_x_down_2_1, x_up_3_3 = self.block_mif_3_3(torch.cat([x_down_2_3, x_3_2, x_up_4_1], dim=1))

        x_plus_3_1 = self.up3_1_relu(self.up3_1_bn(self.up3_1_conv(self.up3_1(torch.cat([x_down_2_3, x_3_2, x_up_4_1], dim=1)))))

        x_1_6 = self.block_mif_1_6(torch.cat([x_1_5, x_up_2_4], dim=1))
        old_x_down_1_1, x_up_2_5 = self.block_mif_2_5(torch.cat([x_plus_4_2,x_down_1_5, x_2_4, x_up_3_3], dim=1))
        x_1_7 = self.block_mif_1_7(torch.cat([x_plus_3_1,x_plus_4_1,x_1_6, x_up_2_5], dim=1))

        #stage 2
        new_x = torch.cat([x,self.final5(x_1_7)], dim=1)
        new_x_1_1, new_x_down_1_1 = self.new_block_mif_1_1(new_x)

        #SFCC
        new_x_plus_4_2 = self.up4_2_relu(self.up4_2_bn(self.up4_2(new_x_down_1_1)))
        new_x_plus_4_1 = self.up4_1_relu(self.up4_1_bn(self.up4_1_conv(self.up4_1(new_x_down_1_1))))


        new_x_1_2, new_x_down_1_2 = self.block_mif_1_2(new_x_1_1)
        #CFCC
        new_x_2_1, new_x_up_2_1, new_x_down_2_1 = self.new_block_mif_2_1(old_x_down_1_1+new_x_down_1_1)
        # new_x_2_1, new_x_up_2_1, new_x_down_2_1 = self.new_block_mif_2_1(new_x_down_1_1)
        new_x_1_3, new_x_down_1_3 = self.block_mif_1_3(torch.cat([new_x_1_2, new_x_up_2_1], dim=1))
        new_x_2_2, new_x_up_2_2, new_x_down_2_2 = self.block_mif_2_2(
            torch.cat([new_x_down_1_2, new_x_2_1], dim=1))
        new_x_3_1, new_x_up_3_1, new_x_down_3_1 = self.new_block_mif_3_1(old_x_down_2_1+ new_x_down_2_1)
        # new_x_3_1, new_x_up_3_1, new_x_down_3_1 = self.new_block_mif_3_1(new_x_down_2_1)
        new_x_1_4, new_x_down_1_4 = self.block_mif_1_4(torch.cat([new_x_1_3, new_x_up_2_2], dim=1))
        new_x_2_3, new_x_up_2_3, new_x_down_2_3 = self.block_mif_2_3(
            torch.cat([new_x_down_1_3, new_x_2_2, new_x_up_3_1], dim=1))
        new_x_3_2, new_x_up_3_2 = self.block_mif_3_2(torch.cat([new_x_down_2_2, new_x_3_1], dim=1))
        _, new_x_up_4_1 = self.new_block_mif_4_1(old_x_down_3_1+new_x_down_3_1)
        # _, new_x_up_4_1 = self.new_block_mif_4_1(new_x_down_3_1)
        new_x_1_5, new_x_down_1_5 = self.block_mif_1_5(torch.cat([new_x_1_4, new_x_up_2_3], dim=1))
        new_x_2_4, new_x_up_2_4 = self.block_mif_2_4(torch.cat([new_x_down_1_4, new_x_2_3, new_x_up_3_2], dim=1))
        _, new_x_up_3_3 = self.block_mif_3_3(torch.cat([new_x_down_2_3, new_x_3_2, new_x_up_4_1], dim=1))

        new_x_plus_3_1 = self.up3_1_relu(self.up3_1_bn(self.up3_1_conv(self.up3_1(torch.cat([new_x_down_2_3, new_x_3_2, new_x_up_4_1], dim=1)))))

        new_x_1_6 = self.block_mif_1_6(torch.cat([new_x_1_5, new_x_up_2_4], dim=1))
        _, new_x_up_2_5 = self.block_mif_2_5(torch.cat([new_x_plus_4_2,new_x_down_1_5, new_x_2_4, new_x_up_3_3], dim=1))
        new_x_1_7 = self.block_mif_1_7(torch.cat([new_x_plus_3_1,new_x_plus_4_1,new_x_1_6, new_x_up_2_5], dim=1))






        if self.out_ave == True:
        #5FDS
            output = (self.final1(new_x_1_3) + self.final2(new_x_1_4) +
                      self.final3(new_x_1_5) + self.final4(new_x_1_6) + self.final5(new_x_1_7)) / 5

        # # 10FDS
        # output = (self.final1(x_1_3) + self.final2(x_1_4) +
        #           self.final3(x_1_5) + self.final4(x_1_6) + self.final5(x_1_7)+
        #           self.final1(new_x_1_3) + self.final2(new_x_1_4) +
        #           self.final3(new_x_1_5) + self.final4(new_x_1_6) + self.final5(new_x_1_7)) / 10
        # # 2FDS
        # output = (self.final5(x_1_7) + self.final5(new_x_1_7)) / 2
        else:
        #NO DS
            output = self.final5(new_x_1_7)

        return output


