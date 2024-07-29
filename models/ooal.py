import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip import clip
from models.coop import TextEncoder, PromptLearner
from models.seg_decoder import SegDecoder


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Net(nn.Module):
    def __init__(self, args, input_dim, out_dim, dino_pretrained='dinov2_vitb14'):
        super().__init__()
        self.dino_pretrained = dino_pretrained
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.class_names = args.class_names
        self.num_aff = len(self.class_names)

        # set up a vision embedder
        self.embedder = Mlp(in_features=input_dim, hidden_features=int(out_dim), out_features=out_dim,
                            act_layer=nn.GELU, drop=0.)
        self.dino_model = torch.hub.load('facebookresearch/dinov2', self.dino_pretrained).cuda()

        clip_model = load_clip_to_cpu('ViT-B/16').float()
        classnames = [a.replace('_', ' ')for a in self.class_names]
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.aff_text_encoder = TextEncoder(clip_model)

        self.seg_decoder = SegDecoder(embed_dims=out_dim, num_layers=2)

        self.merge_weight = nn.Parameter(torch.zeros(3))

        self.lln_linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(3)])
        self.lln_norm = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(3)])

        self.lln_norm_1 = nn.LayerNorm(out_dim)
        self.lln_norm_2 = nn.LayerNorm(out_dim)

        self.linear_cls = nn.Linear(input_dim, out_dim)

        self._freeze_stages(exclude_key=['embedder', 'ctx', 'seg_decoder', 'lln_', 'merge_weight', 'linear_cls'])

    def forward(self, img, label=None, gt_aff=None):
        # {
        #     "x_norm_clstoken": x_norm[:, 0],
        #     "x_norm_patchtokens": x_norm[:, 1:],
        #     "x_prenorm": x,
        #     "masks": masks,
        # }

        b, _, h, w = img.shape

        # Last N features from DINO
        dino_out = self.dino_model.get_intermediate_layers(img, n=3, return_class_token=True)
        merge_weight = torch.softmax(self.merge_weight, dim=0)

        dino_dense = 0
        for i, feat in enumerate(dino_out):
            feat_ = self.lln_linear[i](feat[0])
            feat_ = self.lln_norm[i](feat_)
            dino_dense += feat_ * merge_weight[i]

        dino_dense = self.lln_norm_1(self.embedder(dino_dense))

        # Affordance Text Encoder
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.lln_norm_2(self.aff_text_encoder(prompts, tokenized_prompts))

        dino_cls = dino_out[-1][1]      # b x 768
        dino_cls = self.linear_cls(dino_cls)

        text_features = text_features.unsqueeze(0).expand(b, -1, -1)
        text_features, attn_out, _ = self.seg_decoder(text_features, dino_dense, extra_token=dino_cls)

        attn = (text_features[-1] @ dino_dense.transpose(-2, -1)) * (512 ** -0.5)
        attn_out = torch.sigmoid(attn)
        attn_out = attn_out.reshape(b, -1, h // 14, w // 14)
        pred = F.interpolate(
            attn_out, size=img.shape[-2:], mode="bilinear", align_corners=False
        )

        if self.training:
            assert not label == None, 'Label should be provided during training'
            loss_bce = nn.BCELoss()(pred, label / 255.0)
            loss_dict = {'bce': loss_bce}
            return pred, loss_dict

        else:
            if gt_aff is not None:
                out = torch.zeros(b, h, w).cuda()
                for b_ in range(b):
                    out[b_] = pred[b_, gt_aff[b_]]
                return out

    def _freeze_stages(self, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in self.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count > 0:
                        print('Finetune layer in backbone:', n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False
