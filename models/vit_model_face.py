import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from torch.nn import Parameter
from IPython import embed
import math

MIN_NUM_PATCHES = 16

class Softmax(nn.Module):
    r"""Implement of Softmax (normal classification head):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        """
    def __init__(self, in_features, out_features, device_id):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, label):
        if self.device_id == None:
            out = F.linear(x, self.weight, self.bias)
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            sub_biases = torch.chunk(self.bias, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            bias = sub_biases[0].cuda(self.device_id[0])
            out = F.linear(temp_x, weight, bias)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                bias = sub_biases[i].cuda(self.device_id[i])
                out = torch.cat((out, F.linear(temp_x, weight, bias).cuda(self.device_id[0])), dim=1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()



class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.50, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.to(self.device_id[0]) #one_hot.cuda(self.device_id[0])
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot.scatter_(1, label.cuda(self.device_id[0]).view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m
        print("self.device_id", self.device_id)
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------

        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot

        one_hot.scatter_(1, label.cuda(self.device_id[0]).view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'

class SFaceLoss(nn.Module):

    def __init__(self, in_features, out_features, device_id, s = 64.0, k = 80.0, a = 0.90, b = 1.2):
        super(SFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.k = k
        self.a = a
        self.b = b
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        #nn.init.xavier_uniform_(self.weight)
        xavier_normal_(self.weight, gain=2, mode='out')

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))

            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])

                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)
        # --------------------------- s*cos(theta) ---------------------------
        output = cosine * self.s
        # --------------------------- sface loss ---------------------------

        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1), 1)

        zero_hot = torch.ones(cosine.size())
        if self.device_id != None:
            zero_hot = zero_hot.cuda(self.device_id[0])
        zero_hot.scatter_(1, label.view(-1, 1), 0)


        WyiX = torch.sum(one_hot * output, 1)
        with torch.no_grad():
            theta_yi = torch.acos(WyiX / self.s)
            weight_yi = 1.0 / (1.0 + torch.exp(-self.k * (theta_yi - self.a)))
        intra_loss = - weight_yi * WyiX

        Wj = zero_hot * output
        with torch.no_grad():
            # theta_j = torch.acos(Wj)
            theta_j = torch.acos(Wj / self.s)
            weight_j = 1.0 / (1.0 + torch.exp(self.k * (theta_j - self.b)))
        inter_loss = torch.sum(weight_j * Wj, 1)

        loss = intra_loss.mean() + inter_loss.mean()
        Wyi_s = WyiX / self.s
        Wj_s = Wj / self.s
        return output, loss, intra_loss.mean(), inter_loss.mean(), Wyi_s.mean(), Wj_s.mean()



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        # if isinstance(self.fn, Attention):
        #     attn, out = self.fn(x, **kwargs)
        #     return attn, out + x
        # else:
        attn, out = self.fn(x, **kwargs)
        return attn, out + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        if isinstance(self.fn, Attention):
            attn, out = self.fn(x, **kwargs)
            return attn, out
        else:
            return None, self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        #embed()
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return attn, out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        attns = []
        for attn, ff in self.layers:
            att, x = attn(x, mask = mask)
            if att is not None:
                attns.append(att)
            #embed()
            _, x = ff(x)
        return x, attns

class Hybrid_ViT(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, ac_patch_size,
                         pad, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., 
                         out_dim=512, remove_pos=False):

        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = 8 ** 2 #16**2 #
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.patch_size = patch_size
        self.face_model = None
        self.remove_pos = remove_pos
        if self.remove_pos == False:
            # self.pos_embedding = nn.Parameter(torch.randn(1, 2*num_patches + 2, dim))
            self.pos_embedding = nn.Parameter(torch.randn(1, 2*num_patches + 1, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(out_dim),
        )

        self.loss_type = loss_type
        self.GPU_ID = GPU_ID

        if self.loss_type == 'None':
            print("no loss for vit_face")
        else:
            if self.loss_type == 'Softmax':
                self.loss = Softmax(in_features=out_dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'CosFace':
                self.loss = CosFace(in_features=out_dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'ArcFace':
                self.loss = ArcFace(in_features=out_dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'SFace':
                self.loss = SFaceLoss(in_features=out_dim, out_features=num_class, device_id=self.GPU_ID)

    def forward(self, img, label= None , mask = None):
        out = self.face_model(img)
        x = out['embedding_88']
        N, C, _, _ = x.size()
        x = x.view(N, C, -1).transpose(1, 2)
        
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.remove_pos == False:
            x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x, _ = self.transformer(x, mask)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        emb = self.mlp_head(x)
        if label is not None:
            x = self.loss(emb, label)
            return x#, emb
        else:
            return emb

class ViT_face_model(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, ac_patch_size,
                         pad, dim, depth, heads, mlp_dim, no_face_model=False, use_face_loss=False, use_cls=False, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., 
                         out_dim=512, singleMLP=False, remove_sep=False, remove_pos=False):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        if no_face_model:
            num_patches = (image_size // patch_size) ** 2
        else:
            num_patches = 8 ** 2 #16**2 #
            # num_patches = 7 ** 2 # Resnet 50
        patch_dim = channels * ac_patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size
        self.use_cls = use_cls
        self.face_model = None
        self.singleMLP = singleMLP
        self.no_face_model = no_face_model
        self.use_face_loss = use_face_loss
        self.remove_sep = remove_sep
        self.remove_pos = remove_pos
        if self.remove_sep == False:
            self.sep = nn.Parameter(torch.randn(1, 1, dim))
            if self.remove_pos == False:
                self.pos_embedding = nn.Parameter(torch.randn(1, 2*num_patches + 2, dim))
        else:
            if self.remove_pos == False:
                self.pos_embedding = nn.Parameter(torch.randn(1, 2*num_patches + 1, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        k = 8 # 8: resnet 18
        d = 14 #16 #
        # out_dim = 512 # 512  

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(out_dim),
        )
        
        if self.use_cls:
            self.fc = nn.Linear(dim, 2)
        elif self.no_face_model == False:
            if self.singleMLP:
                self.bn = nn.BatchNorm1d(out_dim)
                self.fc = nn.Linear(k*k*dim, out_dim)
            else:
                self.bn1 = nn.BatchNorm1d(out_dim)
                self.bn2 = nn.BatchNorm1d(out_dim)
                # self.fc = nn.Linear(7*7*dim, 512)
                # self.fc = nn.Linear(8*8*dim, 512)
                self.fc1 = nn.Linear(k*k*dim, out_dim)
                self.fc2 = nn.Linear(k*k*dim, out_dim)
        else:
            self.soft_split = nn.Unfold(kernel_size=(ac_patch_size, ac_patch_size), stride=(self.patch_size, self.patch_size), padding=(pad, pad))
            self.patch_to_embedding = nn.Linear(patch_dim, dim)
            self.bn1 = nn.BatchNorm1d(out_dim)
            self.bn2 = nn.BatchNorm1d(out_dim)
            self.fc1 = nn.Linear(d*d*dim, out_dim)
            self.fc2 = nn.Linear(d*d*dim, out_dim)

        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        # if self.loss_type == 'None':
        #     print("no loss for vit_face")
        # else:
        if self.use_face_loss:
            if self.loss_type == 'Softmax':
                self.loss = Softmax(in_features=out_dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'CosFace':
                self.loss = CosFace(in_features=out_dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'ArcFace':
                self.loss = ArcFace(in_features=out_dim, out_features=num_class, device_id=self.GPU_ID)
            elif self.loss_type == 'SFace':
                self.loss = SFaceLoss(in_features=out_dim, out_features=num_class, device_id=self.GPU_ID)

    def forward(self, img, label= None , mask = None, fea=False, vis=False, heatmap=False):
        # p = self.patch_size
        if self.face_model:
            out = self.face_model(img)
            x = out['embedding_88']
            N, C, _, _ = x.size()
            x = x.view(N, C, -1).transpose(1, 2)
        else:
            x = self.soft_split(img).transpose(1, 2)
            x = self.patch_to_embedding(x)
            N, C, _ = x.size()

        b, n, _ = x.shape
        half = int(N/2)
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = int(b/2))
        k = 1
        if self.remove_sep == False:
            sep = repeat(self.sep, '() n d -> b n d', b=int(b/2))
            splits = torch.split(x, half)
            x = torch.cat((splits[0], sep, splits[1]), dim=1)
            k = 2

        x = torch.cat((cls_tokens, x), dim=1)
        if self.remove_pos == False:
            x += self.pos_embedding[:, :(2*n + k)]
        x = self.dropout(x)

        if vis:
            x, attns = self.transformer(x, mask)
        else:
            x, _ = self.transformer(x, mask)
        N, d, C = x.size()
        if self.use_cls:
            x = x[:, 0, :]
            x = self.mlp_head(x)
            x = self.fc(x)
            if fea:
                x = torch.softmax(x, dim=1)
                max_idx = torch.argmax(x, dim=1).view(-1, 1)
                return float(max_idx)
                # x = x.gather(1, max_idx)  
                # x[max_idx==0] = 1.0 - x[max_idx==0]


            return x

        half = int(d/2)
        # splits = torch.split(x[:, 1:, :], half, dim=1)
        # x1, x2 = splits[0].mean(dim=1), splits[1].mean(dim=1)
        if self.remove_sep:
            x1, x2 = x[:, 1:half+1, :], x[:, (half + 1):d, :]
        else:    
            x1, x2 = x[:, 1:half, :], x[:, (half + 1):d, :]

        embedding1, embedding2 = x1, x2
        f1 = x1.mean(dim = 1) #if self.pool == 'mean' else x1[:, 0]
        f2 = x2.mean(dim = 1) #if self.pool == 'mean' else x2[:, 0]

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        if self.singleMLP:
            x1 = self.fc(x1)
            x1 = self.bn(x1)
            x2 = self.fc(x2)
            x2 = self.bn(x2)
        else:
            # x1 = self.bn10(x1)
            x1 = self.fc1(x1)
            x1 = self.bn1(x1)
            # x1 = self.mlp_head(x1)
            # x2 = self.bn20(x2)
            x2 = self.fc2(x2)
            x2 = self.bn2(x2)
            # x2 = self.mlp_head(x2)

        if heatmap:
            return f1, f2, embedding1, embedding2
            # return x1, x2, embedding1, embedding2

        if fea and vis:
            return x1, x2, attns
        if fea: 
            return x1, x2

        # x1, x2 = self.to_latent(x1), self.to_latent(x2)
        if self.use_face_loss:
            x = torch.cat((x1, x2), dim=0)
            emb = self.mlp_head(x)
            if label is not None:
                x = self.loss(emb, label)
                return x    
            #     return x, emb
            # else:
            #     return emb

        cosine = F.cosine_similarity(x1, x2, dim=-1)
        cosine = torch.clamp(cosine, min=0.0, max=1.0)
        return cosine
