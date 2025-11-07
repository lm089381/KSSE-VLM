import torch
import torch.nn as nn
from dam import DAM
from model.vlci import VLCI
import torch.nn.functional as F
from gm_generator import GuidanceMemoryGenerator, ContextGuidanceNorm

def extractor_patch_features(feature_map, patch_size=1):
    B, C, H, W = feature_map.shape

    if patch_size == 1:
        # patch_features = feature_map.permute(0, 2, 3, 1).reshape(B, -1, C)   #(B, H*W, C)
        patch_features = feature_map.reshape(B, C, -1)  # (B, H*W, C)
    else:
        patches = feature_map.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, patch_size,
                                            patch_size)  # (B, C, num_patches, patch_size, patch_size)
        patch_features = patches.mean(dim=[-1, -2])  # 对每个patch提取特征，在最后两个维度上取均值
        patch_features = patch_features.permute(0, 2, 1)  # (B, num_patches, C)

    return patch_features

class SelfAttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        # 定义 Q, K, V 的线性变换
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # x 形状: [B, V, D] (对avg) 或 [B, V, C', W'*H'] (对wxh展平后)
        B, V, _ = x.shape
        Q = self.query(x)  # [B, V, D]
        K = self.key(x)  # [B, V, D]
        V = self.value(x)  # [B, V, D]

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)  # [B, V, V]
        attn_weights = F.softmax(scores, dim=-1)  # 按行归一化

        # 加权求和
        fused = torch.matmul(attn_weights, V)  # [B, V, D]
        return fused.mean(dim=1)  # 聚合所有视角 [B, D]

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, input, query, pad_mask=None, att_mask=None):
        input = input.permute(1, 0, 2)
        query = query.permute(1, 0, 2)
        embed, att = self.attention(query, input, input, key_padding_mask=pad_mask, attn_mask=att_mask)

        embed = self.normalize(embed + query)
        embed = embed.permute(1, 0, 2)
        return embed, att

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention_image_to_text = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.attention_text_to_image = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, img_features, text_features, cross_embed, pad_mask=None, att_mask=None):
        img_features = img_features.permute(1, 0, 2)
        text_features = text_features.permute(1, 0, 2)
        
        return refined_img, refined_text, att_1, att_2

class PointwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, fwd_dim, dropout=0.0):
        super().__init__()
        self.fwd_layer = nn.Sequential(
            nn.Linear(emb_dim, fwd_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fwd_dim, emb_dim),
        )
        self.normalize = nn.LayerNorm(emb_dim)

    def forward(self, input):
        output = self.fwd_layer(input)
        output = self.normalize(output + input)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.0):
        super().__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.fwd_layer = PointwiseFeedForward(embed_dim, fwd_dim, dropout)

    def forward(self, input, pad_mask=None, att_mask=None):
        emb, att = self.attention(input, input, pad_mask, att_mask)
        emb = self.fwd_layer(emb)
        return emb, att


class GenTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.0, gm_mem_len=3):
        super().__init__()
        self.self_attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.cross_attention = MultiheadAttention(embed_dim, num_heads, dropout)

    def forward(self, input,hv,ht, context=None, pad_mask=None, cross_pad_mask=None, att_mask=None, gm=None):
        
        return output, self_att


class TNN(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.1, num_layers=1,
                 num_tokens=1, num_posits=1, token_embedding=None, posit_embedding=None):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, embed_dim) if not token_embedding else token_embedding
        self.posit_embedding = nn.Embedding(num_posits, embed_dim) if not posit_embedding else posit_embedding
        self.transform = nn.ModuleList(
            [TransformerLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)


    def forward(self, token_index=None, token_embed=None, pad_mask=None, pad_id=-1, att_mask=None):
        for i in range(len(self.transform)):
            final_embed = self.transform[i](final_embed, pad_mask, att_mask)[0]

        return final_embed

class CNN(nn.Module):
    def __init__(self, model, model_type='resnet'):
        super().__init__()
        if 'res' in model_type.lower():
            self.feature = nn.Sequential(*list(model.children())[:-2])
            self.average = list(model.children())[:-1][-1]
        elif 'dense' in model_type.lower():
            modules = list(model.features.children())[:-1]
            self.feature = nn.Sequential(*modules)
            self.average = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError('Unsupported model_type!')

    def forward(self, input):
        avg_features = self.average(feat_sum)
        avg_features = avg_features.view(avg_features.shape[0], -1) #[16,1024]

        return avg_features, feat_sum   

class MVCNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  
        self.attn_fusion =  SelfAttentionFusion(feature_dim=1024)

    def forward(self, input):
        img = input[0] 
        pos = input[1]  
        B, V, C, W, H = img.shape   

        img = img.view(B * V, C, W, H)
        avg, wxh = self.model(img)  
        avg = avg.view(B, V, -1)
        wxh = wxh.view(B, V, wxh.shape[-3], wxh.shape[-2], wxh.shape[-1])

        msk = (pos == -1)
        msk_wxh = msk.view(B, V, 1, 1, 1).float()
        msk_avg = msk.view(B, V, 1).float()
        wxh = msk_wxh * (-1) + (1 - msk_wxh) * wxh  
        avg = msk_avg * (-1) + (1 - msk_avg) * avg

        return avg_features, wxh_features, wxh_l, wxh_f   


# --- Main Moduldes ---
class Classifier(nn.Module):
    def __init__(self, num_topics, num_states, cnn=None, tnn=None,
                 fc_features=2048, embed_dim=128, num_heads=1, dropout=0.1):
        super().__init__()
        self.cnn = cnn
        self.tnn = tnn
        self.img_features = nn.Linear(fc_features, num_topics * embed_dim) if cnn != None else None     #nn.Linear(1024,114*256)
        self.txt_features = MultiheadAttention(embed_dim, num_heads, dropout) if tnn != None else None
        self.cross_features = CrossModalAttention(embed_dim, num_heads, dropout)
        self.topic_embedding = nn.Embedding(num_topics, embed_dim)  #num_topics:14+100=114
        self.state_embedding = nn.Embedding(num_states, embed_dim)  #num_states:2
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.num_topics = num_topics
        self.num_states = num_states
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(1114, 114)
       

    def forward(self, img=None, txt=None, lbl=None, txt_embed=None, pad_mask=None, pad_id=3, threshold=0.5,
                get_embed=False, get_txt_att=False):
        if img != None:
            img_features, wxh_features, wxh_l, wxh_f = self.cnn(img)    
            img_features = self.dropout(img_features)
            wxh_features = self.extractor_patch_features(wxh_features, 1).transpose(1, 2)   #[8,64,1024]

        if txt != None:
            if pad_id >= 0 and pad_mask == None:
                pad_mask = (txt == pad_id)
            txt_features = self.tnn(token_index=txt, pad_mask=pad_mask)

        elif txt_embed != None:
            txt_features = self.tnn(token_embed=txt_embed, pad_mask=pad_mask)

        hv, ht = self.vlci(wxh_features, txt_features, mode='train')

        if img != None and (txt != None or txt_embed != None):
            cross_index = torch.arange(self.num_topics).unsqueeze(0).repeat(img_features.shape[0], 1).to(
                img_features.device)
            memory_index = torch.arange(self.num_states).unsqueeze(0).repeat(img_features.shape[0], 1).to(
                img_features.device)

            img_features, txt_features, att_1, att_2 = self.cross_features(img_features, txt_features, cross_embed)   #torch.Size([8, 114, 256])
        
            mem_emb, att = self.attention(memory_embed, img_features + txt_features)
            if lbl != None:
                emb = self.state_embedding(lbl)
            else:
                emb = self.state_embedding((att[:, :, 1] > threshold).long())

            final_embed = self.normalize(img_features_with_mem + txt_features_with_mem)

        else:
            raise ValueError('img and txt error')

        if get_embed:
            # return att, final_embed + emb, final_embed + emb2
            return att, final_embed, final_embed,hv,ht
        else:
            return att

class Generator(nn.Module):
    def __init__(self, num_tokens, num_posits, embed_dim=128, num_heads=1, fwd_dim=256, dropout=0.1, num_layers=12):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)
        self.posit_embedding = nn.Embedding(num_posits, embed_dim)
        self.transform = nn.ModuleList(
            [GenTransformerLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers)])
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.num_tokens = num_tokens
        self.num_posits = num_posits
    

    def forward(self, source_embed, source_embed2,hv,ht, token_index=None, source_pad_mask=None, target_pad_mask=None,
                max_len=300, top_k=1, bos_id=1, pad_id=3, mode='eye'):

       
        if token_index != None:
            posit_index = torch.arange(token_index.shape[1]).unsqueeze(0).repeat(token_index.shape[0], 1).to(
                token_index.device)
            posit_embed = self.posit_embedding(posit_index)
            token_embed = self.token_embedding(token_index)
            target_embed = token_embed + posit_embed

            final_embed = torch.cat([source_embed, target_embed], dim=1)

            "-----------------------"
            # final_embed = self.activation(self.fuse_layer(final_embed))

            if source_pad_mask == None:
                source_pad_mask = torch.zeros((source_embed.shape[0], source_embed.shape[1]),
                                              device=source_embed.device).bool()
            if target_pad_mask == None:
                target_pad_mask = torch.zeros((target_embed.shape[0], target_embed.shape[1]),
                                              device=target_embed.device).bool()
            pad_mask = torch.cat([source_pad_mask, target_pad_mask], dim=1)
            cross_pad_mask = source_pad_mask
            att_mask = self.generate_square_subsequent_mask_with_source(source_embed.shape[1], target_embed.shape[1],
                                                                        mode).to(final_embed.device)

            for i in range(len(self.transform)):
                # gm = self.gm_module(gm, final_embed[:, -1:, :])  # 更新 GM
                final_embed = self.transform[i](final_embed,hv,ht, source_embed2, pad_mask, cross_pad_mask, att_mask)[0]
            token_index = torch.arange(self.num_tokens).unsqueeze(0).repeat(token_index.shape[0], 1).to(
                token_index.device)
            token_embed = self.token_embedding(token_index)
            emb, att = self.attention(token_embed, final_embed)
            emb = emb[:, source_embed.shape[1]:, :]
            att = att[:, source_embed.shape[1]:, :]
            return att, emb
        else:
            return self.infer(source_embed, source_embed2,hv,ht, source_pad_mask, max_len, top_k, bos_id, pad_id)

    def infer(self, source_embed, source_embed2,hv,ht, source_pad_mask=None, max_len=100, top_k=1, bos_id=1, pad_id=3):
        outputs = torch.ones((top_k, source_embed.shape[0], 1), dtype=torch.long).to(source_embed.device) * bos_id
        scores = torch.zeros((top_k, source_embed.shape[0]), dtype=torch.float32).to(source_embed.device)
        for _ in range(1, max_len):
            possible_outputs = []
            possible_scores = []
            for k in range(top_k):
                output = outputs[k]
                score = scores[k]
                att, emb = self.forward(source_embed, source_embed2,hv,ht, output, source_pad_mask=source_pad_mask,
                                        target_pad_mask=(output == pad_id))
                val, idx = torch.topk(att[:, -1, :], top_k)
                log_val = -torch.log(val)
                for i in range(top_k):
                    new_output = torch.cat([output, idx[:, i].view(-1, 1)], dim=-1)
                    new_score = score + log_val[:, i].view(-1)
                    possible_outputs.append(new_output.unsqueeze(0))
                    possible_scores.append(new_score.unsqueeze(0))
            possible_outputs = torch.cat(possible_outputs, dim=0)
            possible_scores = torch.cat(possible_scores, dim=0)
            val, idx = torch.topk(possible_scores, top_k, dim=0)
            col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0], 1)
            outputs = possible_outputs[idx, col_idx]
            scores = possible_scores[idx, col_idx]
        val, idx = torch.topk(scores, 1, dim=0)
        col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0], 1)
        output = outputs[idx, col_idx]
        return output.squeeze(0)

    def generate_square_subsequent_mask_with_source(self, src_sz, tgt_sz, mode='eye'):
        mask = self.generate_square_subsequent_mask(src_sz + tgt_sz)
        if mode == 'one':
            mask[:src_sz, :src_sz] = self.generate_square_mask(src_sz)
        elif mode == 'eye':
            mask[:src_sz, :src_sz] = self.generate_square_identity_mask(src_sz)
        else:
            raise ValueError('Mode must be "one" or "eye".')
        mask[src_sz:, src_sz:] = self.generate_square_subsequent_mask(tgt_sz)
        return mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_identity_mask(self, sz):
        mask = (torch.eye(sz) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_mask(self, sz):
        mask = (torch.ones(sz, sz) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
