import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import os

from minigpt4.common.registry import registry
from minigpt4.models.rec_model import Rec2Base, disabled_train
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer, GenerationConfig
import re
import numpy as np
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

def get_ids_order(prompt):
    id_flags = ["<UserID>", "<ItemIDList>", "<TargetItemID>"]
    id_order_ = []
    for flag_ in id_flags:
        pos_ = prompt.find(flag_)
        if pos_>=0:
            id_order_.append(pos_)
    id_order_ = np.argsort(np.array(id_order_))
    return id_order_

def consitence_loss(ori_embs, proj_embs):
    ori_embs = ori_embs.squeeze()
    proj_embs = proj_embs.squeeze()
    ori_similarities = torch.matmul(ori_embs, ori_embs.T)
    # ori_diag = torch.diag(ori_similarities)+1e9
    proj_similarities = torch.matmul(proj_embs, proj_embs.T)
    # proj_diag = torch.diag(proj_similarities)+1e9
    N_ = ori_similarities.shape[0]
    ori_similarities[range(N_), range(N_)] -= 1e9
    proj_similarities[range(N_), range(N_)] -= 1e9
    ori_similarities = torch.softmax(ori_similarities,dim=-1) 
    proj_similarities = torch.softmax(proj_similarities,dim=-1)
    loss = nn.functional.mse_loss(ori_similarities, proj_similarities)
    # loss = -torch.log(proj_similarities+1e-6).mul(ori_similarities).sum(dim=-1).mean() #+ nn.functional.cross_entropy(,)
    # loss = nn.functional.kl_div(proj_similarities, ori_similarities, reduction="batchmean")
    return loss 

class identical_map(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self,x):
        return x*1.0


@registry.register_model("mini_gpt4rec_v2")
class MiniGPT4Rec_v2(Rec2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4rec.yaml",
    }

    def __init__(
        self,
        rec_model="MF",
        rec_config=None,
        pretrained_rec=None,
        freeze_rec=True,
        rec_precision='fp16',
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        proj_token_num=1, # the number of tokens that the user/item embedding projected to
        proj_drop=0,
        lora_config=None,
        proj_mid=5,
        freeze_lora=False,
        freeze_proj=False,
        upstream_adapter=None,
        initialization_seed=None
    ):
        super().__init__()

        # self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.proj_token_num = proj_token_num

        print("runing MiniGPT4Rec_v2 ...... ")

        print('Loading Rec_model')
        self.rec_model_type = rec_model
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config, rec_precision)
        # try:
        if self.rec_encoder is not None and pretrained_rec != "not_have":
            self.rec_encoder.load_state_dict(torch.load(pretrained_rec, map_location="cpu"))
            print("successfully load the pretrained model......")
        # except:
        #     # print(pretrained_rec)
        #     # self.rec_encoder.config
        #     raise RuntimeError("Please provide your pretained rec model path or check whether the pretrained model and the defined mode can match each other")
        if freeze_rec and self.rec_encoder is not None:
            for name, param in self.rec_encoder.named_parameters():
                param.requires_grad = False
            self.rec_encoder = self.rec_encoder.eval()
            self.rec_encoder.train = disabled_train
            logging.info("freeze rec encoder")
            print("freeze rec encoder")

        print('Loading Rec_model Done')

            
# ！
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        if upstream_adapter:
            if self.low_resource:
                raise ValueError("Merge upstream adapters before enabling quantized downstream loading.")
            exposed_model = PeftModel.from_pretrained(self.llama_model, upstream_adapter, is_trainable=False)
            self.llama_model = exposed_model.merge_and_unload(safe_merge=True)
        self.llama_model.config.use_cache = False
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        if initialization_seed is not None:
            torch.manual_seed(int(initialization_seed))
        self.use_lora = False
        if lora_config is not None and lora_config.use_lora:
            print("Setting Lora")
            self.use_lora = True
            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.dropout,
                bias="none",
                task_type="CAUSAL_LM"
            ) 
            self.llama_model_lora = get_peft_model(self.llama_model, peft_config)
            print("Setting Lora Done")
        
        if freeze_lora and self.use_lora:
            print("freeze lora...")
            for name, param in self.llama_model_lora.named_parameters():
                param.requires_grad = False

        # self.llama_proj = nn.Linear(
        #     self.rec_encoder.config.embedding_size, self.llama_model.config.hidden_size
        # )
        # self.llama_proj_user = nn.Linear(
        #     self.rec_encoder.config.embedding_size, self.llama_model.config.hidden_size
        # )
        # self.llama_proj_item = nn.Linear(
        #     self.rec_encoder.config.embedding_size, self.llama_model.config.hidden_size
        # )
        
        # for normall 
        
        if self.rec_encoder is not None and 'prompt' not in rec_model:
            print("type:", type(proj_mid), proj_mid)
            self.llama_proj = nn.Sequential(
                nn.Linear(self.rec_encoder.config.embedding_size, self.rec_encoder.config.embedding_size*int(proj_mid)),  # ml100=>5
                nn.ReLU(),
                # nn.Dropout(proj_drop),
                nn.Linear(self.rec_encoder.config.embedding_size*int(proj_mid), self.llama_model.config.hidden_size * self.proj_token_num),
            )
            # self.llama_proj = nn.Linear(self.rec_encoder.config.embedding_size, self.llama_model.config.hidden_size * self.proj_token_num)
        elif self.rec_encoder is not None and rec_model=="personlized_prompt": #'prompt' in rec_model:
            # identical mapping function, i.e., f(x)=x
            print("personalized prompt learning....")
            self.llama_proj = nn.Linear(rec_config.item_num+rec_config.user_num, self.llama_model.config.hidden_size * self.proj_token_num,bias=False) #identical_map()
        elif self.rec_encoder is not None and rec_model=="soft_prompt": #'prompt' in rec_model:
            # identical mapping function, i.e., f(x)=x
            print("soft prompt learning....")
            self.llama_proj = nn.Linear(2, self.llama_model.config.hidden_size * self.proj_token_num,bias=False) #identical_map()
        else:
            self.llama_proj = None

        # for name, para in self.llama_proj.named_parameters():
        #     if "weight" in name:
        #         nn.init.constant_(para,0)
        
        if freeze_proj:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            self.llama_proj = self.llama_proj.eval()
            self.llama_proj.train = disabled_train
            logging.info("!!!! freeze llama_proj...")

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.has_print_prompt=False

        # if prompt_path:
        #     with open(prompt_path, 'r') as f:
        #         raw_prompts = f.read().splitlines()
        #     filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
        #     self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
        #     print('Load {} training prompts'.format(len(self.prompt_list)))
        #     print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        # else:
        #     self.prompt_list = []
        if prompt_path:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                raw_prompts = f.read().splitlines()
            # filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<UserID>" in raw_prompt]
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt List: \n{}'.format(self.prompt_list))
            self.has_pri_decode=False
            self.prompt_list_p = None
        else:
            self.prompt_list = []
            self.prompt_list_p = None

    # def vit_to_cpu(self):
    #     self.ln_vision.to("cpu")
    #     self.ln_vision.float()
    #     self.visual_encoder.to("cpu")
    #     self.visual_encoder.float()

    def to_be_trained(self):
        if self.use_lora:
            return True
        # return True # have lora module, will be trained anyway
        id_terms = ["<UserID>", "<ItemIDList>", "<TargetItemID>", "<DCNFeature>"]
        for prompt in self.prompt_list:
            for id_term in id_terms:
                if id_term in prompt:
                    return True
        ### No ID is used, disable the projection layers
        # self.llama_proj = None
        # for name, param in self.llama_proj.named_parameters():
        #     param.requires_grad = False  
        return False
    
    def set_mode(self, mode):
        '''
        mode \in ['v1','v2',None]
        '''
        self.run_mode_ = mode
    
    def rec_to_cpu(self):
        self.rec_encoder.to("cpu")
        self.rec_encoder.float()
    
    def set_answer_type(self,mode):
        if mode == 'v1':
        # pos_ans = ["The former item.", "The first item.", "The former.", "The first.", "The former one.", "The first one."]
        # neg_ans = ["The latter item.", "The second item.", "The latter.", "The second.", "The latter one.", "The second one."]
            self.pos_ans = ["former"]
            self.neg_ans = ["latter"]
        elif mode == 'v2':
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']
            # self.pos_ans = ['enjoy']
            # self.neg_ans = ['dislike']
            pos_ans_id = self.llama_tokenizer(self.pos_ans[0],add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(self.neg_ans[0],add_special_tokens=False).input_ids[0]
            print("answer token ids: pos:",pos_ans_id, "neg ids:", neg_ans_id)
            
        else:
            raise NotImplementedError("not implement this types of answers")
    def print_prompt(self):
        print('Prompt Pos Example \n{} {} or {}'.format(random.choice(self.prompt_list),self.pos_ans[0],self.neg_ans[0]))


    def encode_recdata_v1(self, sample): # used for stage1
        if self.rec_encoder is None:
            return None, None
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')
        with self.maybe_autocast():
            all_user_embeds, all_items_embeds = self.rec_encoder.computer()
            user_embeds = self.rec_encoder.user_encoder(sample['UserID'],all_users=all_user_embeds).unsqueeze(-2)
            targetItem_embed = self.rec_encoder.item_encoder(sample['PairItemIDs'],all_items=all_items_embeds)
            

            user_embeds_llama = self.llama_proj(user_embeds)
            targetItem_embeds_llama = self.llama_proj(targetItem_embed)
        
        sample_embeds_llama = {
            'User_emb': user_embeds_llama,
            'PairItem_emb': targetItem_embeds_llama,
        }
        sample_atts_llama = None
        return sample_embeds_llama, sample_atts_llama

    def encode_recdata_v2(self, sample, ids_order=None):  # used for stage2
        if self.rec_encoder is None:
            return None, None
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')
        
        with self.maybe_autocast():
            batch_size = sample['UserID'].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            all_user_embeds, all_item_embeds = self.rec_encoder.computer()
            if self.rec_model_type == "sasrec":  # for sasrec, there is no user encoder but just seqs encoder, we take it to get user representation
                user_embeds = self.rec_encoder.seq_encoder(sample['sas_seq']).unsqueeze(-2)
            elif self.rec_model_type == "DCN" or self.rec_model_type == "DIN":
                """
                not really user embeding, but the embedding merged for one sample point
                """
                user_embeds = self.rec_encoder.all_encode(sample['UserID'],sample['TargetItemID'],sample['sas_seq'][:,-10:]).unsqueeze(-2)
            else:
                user_embeds = self.rec_encoder.user_encoder(sample['UserID'], all_users=all_user_embeds).unsqueeze(-2)
            # ***Note: here, for sasrec, item embedding comes form the last layer 
            targetItem_embed = self.rec_encoder.item_encoder(sample['TargetItemID'], all_items=all_item_embeds).unsqueeze(-2)
            
            

            user_embeds_llama = self.llama_proj(user_embeds).reshape(batch_size,-1, self.proj_token_num, hidden_size)
            # if self.rec_encoder !="DCN":
            targetItem_embeds_llama = self.llama_proj(targetItem_embed).reshape(batch_size,-1, self.proj_token_num, hidden_size)
            
            # loss_c = consitence_loss(user_embeds, user_embeds_llama) + consitence_loss(targetItem_embed, targetItem_embeds_llama)
            if 'InteractedItemIDs_pad' in sample.keys() and len(ids_order)==3:
                interactedItem_embeds = self.rec_encoder.item_encoder(sample['InteractedItemIDs_pad'], all_items=all_item_embeds)
                interactedItem_embeds_llama = self.llama_proj(interactedItem_embeds).reshape(batch_size,-1, self.proj_token_num, hidden_size)

                merged_embeds = [user_embeds_llama, interactedItem_embeds_llama, targetItem_embeds_llama]
                merged_embeds = [merged_embeds[k] for k in ids_order]
                merged_embeds = torch.cat(merged_embeds,dim=1)              
                idx_flag = torch.ones_like(sample['InteractedItemIDs_pad'])
                idx_flag = torch.where(sample['InteractedItemIDs_pad']==self.rec_encoder.padding_index, 0, idx_flag) # indx_of_paddded historical items
                # to indicate user_id, his_items_id, target_item_id
                idx_flag = [torch.ones([idx_flag.shape[0],1]).to(idx_flag.device),idx_flag,torch.ones([idx_flag.shape[0],1]).to(idx_flag.device)]
                idx_flag = [idx_flag[k] for k in ids_order]
                idx_flag = torch.cat(idx_flag,dim=1).to(device)
                idx_nopad = torch.nonzero(idx_flag)

            

        
            # atts_user = torch.ones(user_embeds_llama.size()[:-1], dtype=torch.long).to(device)
            # atts_targetItem = torch.ones(targetItem_embeds_llama.size()[:-1], dtype=torch.long).to(device)
            # atts_interactedItem =  torch.ones(interactedItem_embeds_llama.size()[:-1], dtype=torch.long).to(device)
                
                #adding consitence loss
                
                 

                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'InteractedItems_embs': interactedItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'merged_embs': merged_embeds[idx_nopad[:,0],idx_nopad[:,1]].reshape(-1, hidden_size),
                    # 'loss_c': loss_c
                }
            else:
                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'InteractedItems_embs': None,
                    'merged_embs': None,
                    # 'loss_c': loss_c
                }
        sample_atts_llama = None
        # {
        #     'user': atts_user,
        #     'TargetItem': atts_targetItem,
        #     'InteractedItems': atts_interactedItem
        # }
        return sample_embeds_llama, sample_atts_llama

    def recprompt_wrap_v1(self, samples, ori_samples, atts_sample, prompt): # used for stage 1
        if prompt:
            prompt_ori = prompt
            split_symbol = ["<UserID>", "<ItemID>"]
            batch_size = ori_samples['UserID'].shape[0]
            bos = "<s>"
            unk_ = self.llama_tokenizer.unk_token #"<unk>"
            prompt = bos + prompt # add the bos
            prompt = prompt.replace("<UserID>", unk_)
            prompt = prompt.replace("<ItemID>", unk_)
            # interactedItems = samples['InteractedItemTitles']
            prompt_list = []
            
            
            for k in range(batch_size):
                prompt_ = prompt+""
                prompt_list.append(prompt_)
            
            # print(prompt_)
            
            self.llama_tokenizer.padding_side = "left"
            prompts_tokens = self.llama_tokenizer(
            prompt_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(samples['User_emb'].device)
            unk_token_id = self.llama_tokenizer.unk_token_id
            replaced_idx = torch.nonzero(prompts_tokens.input_ids==unk_token_id)
            prompt_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
            if "<UserID>" in prompt_ori and "<ItemID>" in prompt_ori:
                prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]]  = torch.cat([samples['User_emb'], samples['PairItem_emb']],dim=-2).reshape(-1,samples['User_emb'].shape[-1])
            else:
                raise RuntimeError("the pretraining just support one type prompt") 
            return prompt_embeds, prompts_tokens.attention_mask

        
    # def recprompt_wrap_v2(self, samples, ori_samples, atts_sample, prompt): # used for stage 2
    #     if prompt:
    #         prompt_ori = prompt
    #         split_symbol = ["<UserID>", "<ItemIDList>", "<ItemTitleList>", "<TargetItemID>", "<TargetItemTitle>"]
    #         batch_size = ori_samples['UserID'].shape[0]
    #         bos = "<s>"
    #         unk_ = self.llama_tokenizer.unk_token #"<unk>"
    #         unk_ = ".".join([unk_]*self.proj_token_num)
    #         prompt = bos + prompt # add the bos
    #         prompt = prompt.replace("<UserID>", unk_)
    #         prompt = prompt.replace("<TargetItemID>", unk_)

    #         # interactedItems = samples['InteractedItemTitles']
    #         prompt_list = []
            
            
    #         for k in range(batch_size):
    #             prompt_ = prompt+""
    #             # prompt_ = prompt.replace('UserID',unk_)
    #             # item_num = samples['interacted']
    #             if 'InteractedNum' in ori_samples.keys():
    #                 prompt_ = prompt_.replace('<ItemIDList>', ', '.join([unk_]*ori_samples['InteractedNum'][k]))
    #                 prompt_ = prompt_.replace("<ItemTitleList>", ori_samples['InteractedItemTitles'][k])
    #             prompt_ = prompt_.replace("<TargetItemTitle>", ori_samples['TargetItemTitle'][k])
    #             # prompt_ = prompt_.replace("<TargetItemID>", unk_)
    #             # prompt_ += samples['Response'][k]
    #             prompt_list.append(prompt_)
            
    #         if not self.has_print_prompt:
    #             print("prompt example:", random.choice(prompt_list))
    #             self.has_print_prompt = True
            
    #         # print(prompt_list[0])
            
    #         self.llama_tokenizer.padding_side = "left"
    #         prompts_tokens = self.llama_tokenizer(
    #         prompt_list,
    #         return_tensors="pt",
    #         padding="longest",
    #         truncation=True,
    #         max_length=self.max_txt_len,
    #         add_special_tokens=False
    #     ).to(ori_samples['UserID'].device)
    #         unk_token_id = self.llama_tokenizer.unk_token_id
    #         if not self.has_pri_decode:
    #             print("#######prmpt decoded example: ",' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[0])))
    #             self.has_pri_decode = True
                

    #         replaced_idx = torch.nonzero(prompts_tokens.input_ids==unk_token_id)
    #         prompt_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
    #         # prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = samples['merged_embs']
    #         if "<UserID>" in prompt_ori  and "<ItemIDList>" in prompt_ori and  "<TargetItemID>" in prompt_ori:
    #             prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = samples['merged_embs']
    #         elif "<UserID>" in prompt_ori and "<TargetItemID>" in prompt_ori and "<ItemIDList>" not in prompt_ori:
    #             prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = torch.cat([samples['User_emb'], samples['TargetItem_emb']],dim=-2).reshape(-1,samples['User_emb'].shape[-1])
    #         else:
    #             pass 
    #         return prompt_embeds, prompts_tokens.attention_mask


        
    def recprompt_wrap_v2(self, samples, ori_samples, atts_sample, prompt):
        """Insert collaborative vectors at explicit markers, independent of unknown text tokens."""
        batch_size = len(ori_samples["UserID"])
        embedding = self.llama_model.get_input_embeddings()
        device = ori_samples["UserID"].device
        sequences = []
        marker_pattern = r"(<UserID>|<TargetItemID>|<DCNFeature>|<ItemIDList>)"
        for index in range(batch_size):
            text = prompt.replace("<ItemTitleList>", ori_samples["InteractedItemTitles"][index])
            text = text.replace("<TargetItemTitle>", ori_samples["TargetItemTitle"][index])
            bos = torch.tensor([self.llama_tokenizer.bos_token_id], device=device)
            chunks = [embedding(bos)]
            for segment in re.split(marker_pattern, text):
                if segment in ("<UserID>", "<DCNFeature>"):
                    chunks.append(samples["User_emb"][index].to(chunks[0].dtype))
                elif segment == "<TargetItemID>":
                    chunks.append(samples["TargetItem_emb"][index].to(chunks[0].dtype))
                elif segment == "<ItemIDList>":
                    raise ValueError("Use user/target features for the CoLLM-MF and CoLLM-DIN prompts.")
                elif segment:
                    ids = self.llama_tokenizer.encode(segment, add_special_tokens=False)
                    if ids:
                        chunks.append(embedding(torch.tensor(ids, device=device, dtype=torch.long)))
            sequence = torch.cat(chunks, dim=0)
            if len(sequence) > self.max_txt_len:
                raise ValueError("Recommendation prompt exceeds max_txt_len; shorten history or increase the budget.")
            sequences.append(sequence)
        width = max(len(sequence) for sequence in sequences)
        masks, padded = [], []
        for sequence in sequences:
            padding = width - len(sequence)
            padded.append(torch.cat([sequence.new_zeros((padding, sequence.shape[-1])), sequence], dim=0))
            masks.append([0] * padding + [1] * len(sequence))
        return torch.stack(padded), torch.tensor(masks, dtype=torch.long, device=device)





    def forward(self,samples):
        if self.run_mode_ == 'v1':
            return self.forward_v1(samples)
        elif self.run_mode_ == 'v2':
            return self.forward_v2(samples)
        else:
            raise NotImplementedError("None-template version has not been implemtned...")  


    def forward_v1(self, samples):
        # sample = samples["image"]
        samples_encode, atts_samples = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            sample_embeds, atts_samples = self.recprompt_wrap_v1(samples_encode, samples, atts_samples, prompt)

        self.llama_tokenizer.padding_side = "right"



        device = samples['UserID'].device #samples_encode['User_emb'].device

        # ans_ = {1: "The user prefers the former item because its MF embedding is a closer match to the user's MF embedding than that of the latter item.", 
        #         0: "The user prefers the latter item because its MF embedding is a closer match to the user's MF embedding than that of the former item."}

        # pos_ans = ["The former item.", "The first item.", "The former.", "The first.", "The former one.", "The first one.",]
        # neg_ans = ["The latter item.", "The second item.", "The latter.", "The second.", "The latter one.", "The second one."]
        # pos_ans = ["The first item."]
        # neg_ans = ["The second item."]
        ans_ = {1: self.pos_ans, 
                0: self.neg_ans}
        text = [random.choice(ans_[int(t)]) + self.end_sym for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0],atts_samples.shape[1]],dtype=torch.long).to(device).fill_(-100)

        # empty_targets = (
        #     torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
        #                dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        # )
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            if self.use_lora:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                )
        loss = outputs.loss

        return {"loss": loss}
    
    def prompt_based_encode_v2(self,prompt, samples):
        id_orders = get_ids_order(prompt)
        samples_encode, atts_samples = self.encode_recdata_v2(samples,ids_order=id_orders)
        sample_embeds, atts_samples = self.recprompt_wrap_v2(samples_encode, samples, atts_samples, prompt)
        return sample_embeds, atts_samples
        

    def prompt_with_p(self,p):
        if self.prompt_list_p is None:
            prompt_list_p= []
            for k in range(len(p)):
                prompt_list_p.extend([self.prompt_list[k]]*p[k])
            self.prompt_list_p = prompt_list_p
            return self.prompt_list_p
        else:
            return self.prompt_list_p


    def forward_v2(self, samples):
        prompt = random.choice(self.prompt_list)
        logits = self._recommendation_logits(samples, prompt)
        loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), samples["label"].float())
        return {"loss": loss}

    def _recommendation_logits(self, samples, prompt):
        embeddings, attention_mask = self.prompt_based_encode_v2(prompt, samples)
        model = self.llama_model_lora if self.use_lora else self.llama_model
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        with self.maybe_autocast():
            outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask,
                            position_ids=position_ids, return_dict=True, use_cache=False)
        token_ids = self.llama_tokenizer.encode(self.pos_ans[0], add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError("Positive answer must tokenize to one token for recommendation scoring.")
        return outputs.logits[:, -1, token_ids[0]]




# sample_embeds_llama = {
#                     'User_emb': user_embeds_llama.reshape(batch_size,-1, hidden_size),
#                     'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size,-1, hidden_size),
#                     'InteractedItems_embs': None,
#                     'merged_embs': None
#                 }


    def generate_for_samples_v2(self, samples, return_all=False):
        logits = self._recommendation_logits(samples, self.prompt_list[0])
        loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), samples["label"].float())
        if return_all:
            return {"logits": logits}, logits
        return {"loss": loss, "logits": logits}

    



    # def generate_for_samples_v2_new(self, samples,return_all=False):
    #     # sample = samples["image"]
        
    #     if hasattr(samples, 'question_split'):  # VQA dataset
    #         print('VQA Batch')
    #         raise NotImplementedError("not implement")
    #         # vqa_prompt = '###Human: <Img><ImageHere></Img> '
    #         # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
    #     elif self.prompt_list:
    #         # prompt = random.choice(self.prompt_list)
    #         prompt = self.prompt_list[0] # default 0
    #         id_orders = get_ids_order(prompt)
    #         samples_encode, atts_samples = self.encode_recdata_v2(samples,ids_order=id_orders)
    #         sample_embeds, atts_samples = self.recprompt_wrap_v2(samples_encode, samples, atts_samples, prompt)

    #     self.llama_tokenizer.padding_side = "right"



    #     device = samples['UserID'].device #samples_encode['User_emb'].device
    #     inputs_embeds = sample_embeds
    #     attention_mask = atts_samples


    #     with self.maybe_autocast():
    #         if not self.use_lora:
    #             outputs = self.llama_model(
    #                 inputs_embeds=inputs_embeds,
    #                 attention_mask=attention_mask,
    #                 return_dict=True,
    #                 labels=None,
    #             )
    #         else:
    #             outputs = self.llama_model_lora(
    #                 inputs_embeds=inputs_embeds,
    #                 attention_mask=attention_mask,
    #                 return_dict=True,
    #                 labels=None,
    #             )
    #     # loss = outputs.loss
    #     pos_ans = self.pos_ans[0]
    #     neg_ans = self.neg_ans[0]
    #     ans_ = {1:pos_ans, 0:neg_ans}
    #     pos_ans_id = self.llama_tokenizer(pos_ans, add_special_tokens=False).input_ids[0]
    #     neg_ans_id = self.llama_tokenizer(neg_ans, add_special_tokens=False).input_ids[0]

    #     # logits = outputs.logits[:,-2,:][:,[neg_ans_id, pos_ans_id]]
    #     # logits_ = torch.ones_like(logits[:,0]).float()
    #     # logits_ = torch.where(logits[:,0]>logits[:,1],0,logits_)
    #     # logits_ = torch.where(logits[:,0]==logits[:,1],0.5,logits_)

    #     logits_ = outputs.logits[:,-1,:][:, pos_ans_id]
    #     # print(lo)
    #     loss = nn.functional.binary_cross_entropy_with_logits(logits_, samples['label'].float())

    #     if return_all:
    #         return outputs, logits_


    #     return {"loss": loss, 'logits':logits_}


    def generate_sequence(self,samples):
        
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
            # vqa_prompt = '###Human: <Img><ImageHere></Img> '
            # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            id_orders = get_ids_order(prompt)
            samples_encode, atts_samples = self.encode_recdata_v2(samples,ids_order=id_orders)
            sample_embeds, atts_samples = self.recprompt_wrap_v2(samples_encode, samples, atts_samples, prompt)

        inputs_embeds =  sample_embeds #torch.cat([sample_embeds, to_regress_embeds], dim=1)
        with torch.no_grad():
            try:
                if not self.use_lora:
                    outputs = self.llama_model.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=10,
                    num_beams=1,
                    do_sample=True,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    length_penalty=1,
                    temperature=1.0,
                    return_dict_in_generate=True,
                    output_scores=True)
                else:
                    outputs = self.llama_model_lora.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=10,
                    num_beams=1,
                    do_sample=True,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    length_penalty=1,
                    temperature=1.0,
                    return_dict_in_generate=True,
                    output_scores=True)
            except:
                print("errors.....")
        print(inputs_embeds.shape, outputs.sequences.shape)
        print(self.llama_tokenizer.batch_decode(outputs.sequences,skip_special_tokens=True), samples['label']) 
        print()
        return {"loss": 0, 'logits':outputs.logit}
        # return outputs

    def encode_allinputs(self,samples,mode='v1'):
        if mode=='v2':
            samples_encode, atts_samples = self.encode_recdata_v2(samples)
        else:
            samples_encode, atts_samples = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
            # vqa_prompt = '###Human: <Img><ImageHere></Img> '
            # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            if mode=='v2':
                sample_embeds, atts_samples = self.recprompt_wrap_v2(samples_encode, samples, atts_samples, prompt)
            else:
                sample_embeds, atts_samples = self.recprompt_wrap_v1(samples_encode, samples, atts_samples, prompt)

        inputs_embeds =  sample_embeds #torch.cat([sample_embeds, to_regress_embeds], dim=1)
        return inputs_embeds
        



    
    def generate_for_samples_v1(self, samples):
        
        
        samples_encode, atts_samples = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            raise NotImplementedError("not implement")
            # vqa_prompt = '###Human: <Img><ImageHere></Img> '
            # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            sample_embeds, atts_samples = self.recprompt_wrap_v1(samples_encode, samples, atts_samples, prompt)

        self.llama_tokenizer.padding_side = "right"



        device = samples_encode['User_emb'].device
        # sample = samples["image"]
        # ans_ = {1: "The user prefers the former item because its MF embedding is a closer match to the user's MF embedding than that of the latter item.", 
        #         0: "The user prefers the latter item because its MF embedding is a closer match to the user's MF embedding than that of the former item."}
        # pos_ans = ["The former item.", "The first item.", "The former.", "The first.", "The former one.", "The first one."]
        # neg_ans = ["The latter item.", "The second item.", "The latter.", "The second.", "The latter one.", "The second one."]
        # pos_ans = ["The first item."]
        # neg_ans = ["The second item."]
        # pos_ans = ["The first item."]
        # neg_ans = ["The second item."]
        ans_ = {1: self.pos_ans, 
                0: self.neg_ans}
        text = [random.choice(ans_[int(t)]) + self.end_sym for t in samples["label"]]

        # text = [ans_[int(t)] + self.end_sym for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0],atts_samples.shape[1]],dtype=torch.long).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            if not self.use_lora:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        loss = outputs.loss
        return {"loss": loss}
    

    def generate_for_samples(self,samples):
        if self.run_mode_ == 'v1':
            return self.generate_for_samples_v1(samples)
        elif self.run_mode_ == 'v2':
            return self.generate_for_samples_v2(samples)
        else:
            raise NotImplementedError("Not implement the default version")     
        # self.generate_sequence(samples)
        # return {'loss':loss, "logits": logits_}

    @classmethod
    def from_config(cls, cfg):
        # rec_model="MF",
        # embedding_size=64,
        # freeze_rec=True,
        # rec_precision='fp16',
        # rec_config = None,
        # llama_model="",
        # prompt_path="",
        # prompt_template="",
        # max_txt_len=32,
        # end_sym='\n',
        # low_resource=False,  # use 8 bit and put vit in cpu
        # device_8bit=0,  # the device of 8bit 


        rec_model = cfg.get('rec_model',"MF")
        rec_config = cfg.rec_config
        embedding_size = cfg.get("rec_emb_size")
        freeze_rec = cfg.get("freeze_rec",True)
        rec_precision = cfg.get("rec_precision", 'fp16')
        rec_config = cfg.get("rec_config")
        lora_config = cfg.get("lora_config")
        llama_model = cfg.get("llama_model")
        proj_token_num = cfg.get("proj_token_num")
        proj_mid = cfg.get("proj_mid_times")
        freeze_proj = cfg.get("freeze_proj")
        freeze_lora = cfg.get("freeze_lora")


        # drop_path_rate = cfg.get("drop_path_rate", 0)
        # use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        # vit_precision = cfg.get("vit_precision", "fp16")
        # freeze_vit = cfg.get("freeze_vit", True)
        # freeze_qformer = cfg.get("freeze_qformer", True)


        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
             rec_model=rec_model,
             rec_config=rec_config,
             pretrained_rec = rec_config['pretrained_path'],
             freeze_rec=freeze_rec,
             rec_precision=rec_precision,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            proj_token_num = cfg.get("proj_token_num"),
            proj_drop = cfg.get("proj_drop"),
            lora_config = lora_config,
            proj_mid = proj_mid,
            freeze_lora=freeze_lora,
            freeze_proj=freeze_proj,
            upstream_adapter=cfg.get("upstream_adapter"),
            initialization_seed=cfg.get("initialization_seed")
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load MiniGPT4Rec Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            # msg = model.load_state_dict(ckpt['model'], strict=False)
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print("loading message, msg.... ", msg)
            # reload the rec model, avoiding it be covered by the loaded ckpt
            if os.path.exists(rec_config['pretrained_path']) and freeze_rec:
                model.rec_encoder.load_state_dict(torch.load(rec_config['pretrained_path'], map_location="cpu"))
        ans_type = cfg.get('ans_type')
        model.set_answer_type(mode=ans_type)
        model.print_prompt()
        return model
