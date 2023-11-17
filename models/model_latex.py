class H2L(nn.Module):
    def __init__(self, *, num_class, image_size, patch_size, ac_patch_size,
                         pad, dim, depth, heads, mlp_dim, resnet_model, channels = 3, 
                         dim_head = 64, dropout = 0., emb_dropout = 0., out_dim=512):
        super().__init__()
        num_patches = 8 ** 2 
        patch_dim = channels * ac_patch_size ** 2

        self.resnet_model = None
        self.sep = nn.Parameter(torch.randn(1, 1, dim))    
        self.pos_embedding = nn.Parameter(torch.randn(1, 2*num_patches + 2, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(out_dim),
        )
        
       
        self.soft_split = nn.Unfold(kernel_size=(ac_patch_size, ac_patch_size), stride=(self.patch_size, self.patch_size), padding=(pad, pad))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.fc1 = nn.Linear(d*d*dim, out_dim)
        self.fc2 = nn.Linear(d*d*dim, out_dim)
        
        self.loss = ArcFace(in_features=out_dim, out_features=num_class)

    def forward(self, img, label= None , mask = None):
        out = self.resnet_model(img)
        x = out['embedding_88']
        N, C, _, _ = x.size()
        x = x.view(N, C, -1).transpose(1, 2)

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
        x = self.dropout(x)

        x, attns = self.transformer(x, mask)
        N, d, C = x.size()

        half = int(d/2)
        x1, x2 = x[:, 1:half, :], x[:, (half + 1):d, :]
        embedding1, embedding2 = x1, x2
        f1 = x1.mean(dim = 1) 
        f2 = x2.mean(dim = 1)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x1 = self.mlp_head(x1)
        x1 = self.fc1(x1)
        x1 = self.bn1(x1)

        x2 = self.mlp_head(x2)
        x2 = self.fc2(x2)
        x2 = self.bn2(x2)

        x = torch.cat((x1, x2), dim=0)
        emb = self.mlp_head(x)
        x = self.loss(emb, label)
        return x    


class H2(nn.Module):
    def __init__(self, *, num_class, image_size, patch_size, ac_patch_size,
                          pad, dim, depth, heads, mlp_dim, resnet_model, channels = 3, 
                          dim_head = 64, dropout = 0., emb_dropout = 0., out_dim=512):
        super().__init__()
        num_patches = 8 ** 2
        patch_dim = channels * ac_patch_size ** 2
        self.patch_size = patch_size
        self.resnet_model = resnet_model # # resnet to extract embeedings
        self.sep = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 2*num_patches + 2, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity() 
        self.ln = nn.LayerNorm(out_dim)
        self.fc = nn.Linear(dim, 2)  # outputs

    def forward(self, img, label= None , mask = None):
        if self.face_model:
        out = self.resnet_model(img)
        x = out['embedding_88']
        N, C, _, _ = x.size()
        x = x.view(N, C, -1).transpose(1, 2)

        b, n, _ = x.shape
        half = int(N/2)
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = int(b/2))
        sep = repeat(self.sep, '() n d -> b n d', b=int(b/2))
        splits = torch.split(x, half)
        x = torch.cat((splits[0], sep, splits[1]), dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(2*n + 2)]
        x = self.dropout(x)
        x, attns = self.transformer(x, mask)
        N, d, C = x.size()
        x = x[:, 0, :]
        x = self.mlp_head(x)
        x = self.fc(x)
        return x