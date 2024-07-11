import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from .invpt import InvPT
from einops import rearrange as o_rearrange
def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()
class MLPHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MLPHead, self).__init__()

        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.linear_pred(x)


# class ConvBlock(nn.Module):
#     def __init__(self, inplanes, planes,  norm_layer=None):
#         super(ConvBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.SyncBatchNorm # nn.BatchNorm2d
#         self.conv = nn.Conv2d(inplanes, planes, 3, padding=1)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         return out
def get_head(num_outputs,backbone_channels, task):
    """ Return the decoder head """

    return MLPHead(backbone_channels, num_outputs[task])
class MultiTaskSwin(nn.Module):
    def __init__(self, encoder, config):
        super(MultiTaskSwin, self).__init__()
        self.backbone = encoder
        self.num_outputs = config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT
        self.tasks = config.TASKS
        if hasattr(self.backbone, 'patch_embed'):
            patches_resolution = self.backbone.patch_embed.patches_resolution
            self.embed_dim = self.backbone.embed_dim
            num_layers = self.backbone.num_layers
            self.dims = [int((self.embed_dim * 2 ** ((i+1) if i < num_layers - 1 else i)))
                         for i in range(num_layers)]
            self.input_res = [patches_resolution[0] //
                              (2 ** ((i+1) if i < num_layers - 1 else i)) for i in range(num_layers)]
            self.window_size = self.backbone.layers[0].blocks[0].window_size
            self.img_size = self.backbone.patch_embed.img_size
        else:
            self.input_res = [28, 14, 7, 7]

            self.dims = [192, 384, 768, 768]
            self.window_size = config.MODEL.SWIN.WINDOW_SIZE
            self.img_size = config.DATA.IMG_SIZE

        ######################################
        self.num_prompt=config.MODEL.PROMPT.NUM_TOKENS
        spec = {
            'ori_embed_dim': self.embed_dim,
            'NUM_STAGES': 3,
            'PATCH_SIZE': [0, 3, 3],
            'PATCH_STRIDE': [0, 1, 1],
            'PATCH_PADDING': [0, 2, 2],  #'PATCH_PADDING': [0, 2, 2],
            'DIM_EMBED': [self.embed_dim, self.embed_dim // 2, self.embed_dim // 4],
            'NUM_HEADS': [2, 2, 2],
            'MLP_RATIO': [4., 4., 4.],
            'DROP_PATH_RATE': [0.15, 0.15, 0.15],
            'QKV_BIAS': [True, True, True],
            'KV_PROJ_METHOD': ['avg', 'avg', 'avg'],
            'KERNEL_KV': [2, 4, 8],
            'PADDING_KV': [0, 0, 0],
            'STRIDE_KV': [2, 4, 8],
            'Q_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
            'KERNEL_Q': [3, 3, 3],
            'PADDING_Q': [1, 1, 1],
            'STRIDE_Q': [2, 2, 2],
        }
        self.invpt = InvPT(self.tasks, in_chans=self.embed_dim, spec=spec)
        self.scale_embed = nn.ModuleList()

        self.scale_embed.append(nn.Conv2d(192, spec['DIM_EMBED'][2],3,padding=1))
        self.scale_embed.append(nn.Conv2d(384, spec['DIM_EMBED'][1], 3, padding=1))
        self.scale_embed.append(nn.Conv2d(768, spec['DIM_EMBED'][0], 3, padding=1))
        self.scale_embed.append(None)

        self.conv1=nn.Conv2d(self.dims[-1], self.embed_dim, 1)
        # self.scale_embed_task = nn.ModuleDict()
        # for t in self.tasks:
        #     self.scale_embed_task[t] = nn.Conv2d(768, spec['DIM_EMBED'][0], 3, padding=1)


        self.heads = torch.nn.ModuleDict({task: get_head(self.num_outputs, self.embed_dim, task) for task in self.tasks})
        ######################################

    def forward(self, x):
        shared_representation = self.backbone(x, return_stages=True) 
        #layer1이후 (1,3136,192) ,(1,784,384),(1,196,768),(1,196,768) dim=1에서 +5 swin을 받는다....
        spatial_dim=[
            [self.img_size[0]//8,self.img_size[1]//8],
            [self.img_size[0]//16,self.img_size[1]//16],
            [self.img_size[0]//32,self.img_size[1]//32],
            [self.img_size[0]//32,self.img_size[1]//32]
        ]

        back_fea = []
        for sca in range(len(shared_representation)):
            oh, ow = spatial_dim[sca]
            _fea = shared_representation[sca]

            ##promt제거 prompt,patch
            _fea = _fea[:,self.num_prompt:, :]
            _fea = rearrange(_fea, 'b (h w) c -> b c h w', h=oh, w=ow)
            if sca == 3:
                x = _fea  # use last scale feature as input of InvPT decoder
            if self.scale_embed[sca] != None:
                _fea = self.scale_embed[sca](_fea)
            back_fea.append(_fea)

        #h, w = (self.img_size[0]//32,self.img_size[1]//32) ##self.p.mtt_resolution
        #x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        feat_dict = {}
        _x = self.conv1(x)  #scale dim
        for task in self.tasks :
            #_x=self.scale_embed_task[task](x)
            feat_dict[task] = _x

        x_dict = self.invpt(feat_dict, back_fea)  # multi-scale input
        ##head
        for t in self.tasks: 
            x_dict[t] = F.interpolate(self.heads[t](x_dict[t]),self.img_size, mode='bilinear')
        return x_dict

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_task(self, task):
        for param in self.decoders[task].parameters():
            param.requires_grad = False

    def unfreeze_task(self, task):
        for param in self.decoders[task].parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
