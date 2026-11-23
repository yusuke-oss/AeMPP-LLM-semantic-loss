"""
TPP-LLM Model
"""
import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from peft import get_peft_model, PeftConfig, PeftModel 
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AddedToken

# Local Imports
from src.tpp_llm.taxi.layers import (
    TimePositionalEncoding, 
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



class TPPLLMModel(nn.Module):
    """
    TPP-LLM Model
    """

    def __init__(
        self, model_name: str, num_event_types: int, num_integral_samples: int, temporal_emb_type: str,
        temporal_emb_first: bool = False, prompt: str = '', bnb_config: BitsAndBytesConfig = None,
        peft_config: PeftConfig = None, device: Union[str, torch.device] = 'cpu',
        model_weight_path:str ='',save_flag:bool=False,load_flag:bool=False,train_flag:bool=False,# ★追加: 事前学習済み重みのパスを受け取る引数
        flozen_flag:bool=False,use_natural_separators:bool=False,use_tokens:bool=False, emb_flozen_flag: bool = False,
        codebook_loss_weight_p: float = 0.01,codebook_loss_weight_h: float = 0.01 ,**kwargs):
        """
        Initialize the TPP-LLM model

        :param model_name: LLM name
        :param num_event_types: number of event types
        :param num_integral_samples: number of samples in the intensity integral
        :param temporal_emb_type: temporal embedding type
        :param temporal_emb_first: temporal embedding first (before the text embedding) for each event
        :param prompt: prompt before the event sequences
        :param bnb_config: bits and bytes configuration
        :param peft_config: PEFT configuration
        :param device: device
        """
        super().__init__()
        self.use_natural_separators = use_natural_separators
        self.flozen_flag=flozen_flag
        self.use_tokens=use_tokens
        self.emb_flozen_flag=emb_flozen_flag
        self.temporal_emb_type=temporal_emb_type
        self.codebook_loss_weight_p = codebook_loss_weight_p  # 🌟 【追加】インスタンス変数に保持
        self.codebook_loss_weight_h = codebook_loss_weight_h  # 🌟 【追加】インスタンス変数に保持


        if save_flag or train_flag:
            # Load the LLM
            self.device = torch.device(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.llm = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float32,
                device_map=self.device,
            )
            if flozen_flag:
                for param in self.llm.parameters():
                    param.requires_grad=False
            

            # Apply the PEFT config
            if peft_config is not None:
                self.llm = get_peft_model(self.llm, peft_config)
                self.llm.train()
                self.llm.print_trainable_parameters()
            else:
                self.llm.eval()
                for param in self.llm.parameters():
                    param.requires_grad = False
            
            self.old_vocab_size = self.llm.get_input_embeddings().num_embeddings
            # Added: Flag to disable tokens when using TPE (positional)
            
            
            # Add special event tokens
            if self.use_tokens:
                # ★追加: 自然言語トークンを使わない（特殊トークンを使う）場合のみ辞書に追加！
                if not self.use_natural_separators:
                    event_tokens = [
                        AddedToken("<|time_prefix|>", special=True, lstrip=False, rstrip=False),
                        AddedToken("<|type_prefix|>", special=True, lstrip=False, rstrip=False),
                        AddedToken("<|latitude_prefix|>", special=True, lstrip=False, rstrip=False),
                        AddedToken("<|longitude_prefix|>", special=True, lstrip=False, rstrip=False), 
                        AddedToken("<|distance_prefix|>", special=True, lstrip=False, rstrip=False),  
                    ]
                    self.tokenizer.add_tokens(event_tokens, special_tokens=True)
                    self.llm.resize_token_embeddings(len(self.tokenizer))
                if self.temporal_emb_type=='float':
                    byte_tokens = [
                        AddedToken(f"<|byte_{i}|>", special=True, lstrip=False, rstrip=False)
                        for i in range(256)
                    ]
                    self.tokenizer.add_tokens(byte_tokens, special_tokens=True)
                    self.llm.resize_token_embeddings(len(self.tokenizer))
            
            
                print(f"[TPPLLMModel] old_vocab_size={getattr(self,'old_vocab_size', None)}, "
                    f"total_vocab={len(self.tokenizer)}, "
                    f"embedding_dim={self.llm.get_input_embeddings().embedding_dim}")
            
            # Allow gradients only for the newly added token embeddings
                try:
                    self.llm_embedder = self.llm.get_input_embeddings()
                    self.llm_embedder.weight.requires_grad = True
                except Exception as e:
                    print("Warning: failed to set embedding requires_grad, exception:", e)
            self.llm_embedder = self.llm.get_input_embeddings()
            
            
        if load_flag:
            self.device = torch.device(device)
            print(f"Loading model from {model_weight_path}...")

            # Load tokenizer and add special tokens
            self.tokenizer = AutoTokenizer.from_pretrained(model_weight_path + '/tokenizer')
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # ✨【変更】LoRAの設定ファイルがあるかチェックする
            adapter_config_path = os.path.join(model_weight_path, 'model', 'adapter_config.json')
            is_peft_model = os.path.exists(adapter_config_path)

            if is_peft_model:
                # LoRAを使用している場合
                self.config = PeftConfig.from_pretrained(model_weight_path + '/model')
                base_model_name_to_load = self.config.base_model_name_or_path
            else:
                # 完全に凍結（LoRAなし）の場合、元のモデル名を使う
                base_model_name_to_load = model_name
    
            # Load base model configuration
            self.llm = AutoModel.from_pretrained(
                base_model_name_to_load,
                quantization_config=bnb_config,
                torch_dtype=torch.float32,
                device_map=self.device,
            )
            if flozen_flag:
                for param in self.llm.parameters():
                    param.requires_grad=False

            # Resize token embeddings and restore weights for new tokens
            self.llm.resize_token_embeddings(len(self.tokenizer))
            token_weights_path = os.path.join(model_weight_path, 'new_token_weights.pt')
    
            if os.path.exists(token_weights_path):
                print(f"Loading trained token weights from {token_weights_path}...")
                new_token_weights = torch.load(token_weights_path, map_location=self.device)
                input_embeddings = self.llm.get_input_embeddings()
                num_new_tokens = new_token_weights.shape[0]
        
                with torch.no_grad():
                    input_embeddings.weight.data[-num_new_tokens:] = new_token_weights
                print("Token embeddings updated successfully.")
            else:
                print("Warning: new_token_weights.pt not found! New tokens are randomly initialized.")

            # ✨【変更】LoRAモデルの場合だけアダプタを結合する
            if is_peft_model:
                self.llm = PeftModel.from_pretrained(
                    self.llm,
                    model_weight_path + '/model',
                    torch_dtype=torch.float32,
                    device_map=self.device,
                )
                print("LoRA adapter loaded and merged successfully.")
            else:
                print("No LoRA adapter found. Running with standard LLM (Frozen mode).")
            
            self.llm.eval()
            for param in self.llm.parameters():
                param.requires_grad = False
            print("▶️ Load Mode: 評価用にモデルを凍結しました。")
        
            print("Model loaded and merged successfully.")
            self.old_vocab_size = self.llm.get_input_embeddings().num_embeddings
            self.llm_embedder = self.llm.get_input_embeddings()


        # Set the model parameters
        self.hidden_size = self.llm.config.hidden_size
        self.embedding_dim = self.llm_embedder.embedding_dim
        self.num_integral_samples = num_integral_samples
        self.num_event_types = num_event_types
        self.temporal_emb_first = temporal_emb_first
        self.dtype = self.llm.dtype
        self.prompt = prompt

        
        # Creat layers for computing intensities
        self.intensity_current = nn.Linear(
            1, self.num_event_types, bias=True, dtype=self.dtype, device=self.device)
                        
        # 🌟 履歴依存の強度計算部分をMLP化
        self.intensity_history = nn.Sequential(
            nn.Linear(self.hidden_size, 256, bias=True, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, self.num_event_types, bias=False, dtype=self.dtype, device=self.device)
        )
                
        print('Hiddensize', self.hidden_size)
        self.softplus = nn.Softplus()
        
        # 🌟 イベントタイプ予測ヘッドをMLP化 (中間次元 256)
        self.head_type = nn.Sequential(
            nn.Linear(self.hidden_size, 256, bias=True, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, self.num_event_types, bias=True, dtype=self.dtype, device=self.device),
            nn.Softmax(dim=-1)
        )
                
                # 🌟 イベント時間予測ヘッドをMLP化 (中間次元 256)
        self.head_time = nn.Sequential(
            nn.Linear(self.hidden_size, 256, bias=True, dtype=self.dtype, device=self.device),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1, bias=True, dtype=self.dtype, device=self.device)
        )
        
       
        if self.temporal_emb_type in ['linear', 'linear_flozen']:
            # シンプルな線形変換
            self.time_mlp = nn.Linear(1, self.embedding_dim).to(dtype=self.dtype, device=self.device)
            self.lat_mlp = nn.Linear(1, self.embedding_dim).to(dtype=self.dtype, device=self.device)
            self.lon_mlp = nn.Linear(1, self.embedding_dim).to(dtype=self.dtype, device=self.device)
            self.dis_mlp = nn.Linear(1, self.embedding_dim).to(dtype=self.dtype, device=self.device)
            if emb_flozen_flag:
                # 3つのMLPをリストにまとめ、順番に取り出して処理する
                for mlp in [self.time_mlp, self.lat_mlp, self.lon_mlp, self.dis_mlp]:
                    for param in mlp.parameters():  # ← parameters() と括弧をつけるのを忘れずに！
                        param.requires_grad = False
        
        elif self.temporal_emb_type in ['positional','main']:
            self.time_mlp=TimePositionalEncoding(
                embedding_dim=self.embedding_dim, dtype=self.dtype, device=self.device)
            self.lat_mlp=TimePositionalEncoding(
                embedding_dim=self.embedding_dim, dtype=self.dtype, device=self.device)
            self.lon_mlp=TimePositionalEncoding(
                embedding_dim=self.embedding_dim, dtype=self.dtype, device=self.device)
            self.dis_mlp=TimePositionalEncoding(
                embedding_dim=self.embedding_dim, dtype=self.dtype, device=self.device)

        # 🌟 【変更】MLP_token_renew_loss 系の設定と Codebook の登録
        elif self.temporal_emb_type in ['MLP_token_renew_loss_ph_label','MLP_token_renew_loss_ph', 'MLP_token_renew_topk_loss_ph']:
            if self.temporal_emb_type in ['MLP_token_renew_loss_ph_label']:
                input_dim = 4 
            else:
                input_dim = 4 + self.num_event_types
            self.fusion_mlp = nn.Sequential(
                nn.Linear(input_dim, 256), 
                nn.ReLU(), 
                nn.Dropout(p=0.1),  # ここに挿入（まずは p=0.1 程度の小さめの値から）
                nn.Linear(256, self.embedding_dim)
            ).to(dtype=self.dtype, device=self.device)
            if emb_flozen_flag:
                for param in self.fusion_mlp.parameters():
                    param.requires_grad = False
            
            if self.temporal_emb_type in [
                'MLP_token_renew_loss_ph',
                'MLP_token_renew_topk_loss_ph',
                'MLP_token_renew_loss_ph_label'
            ]:
                # --- プロンプトのベクトル化 ---
                print(f"Initializing Prompt Codebook for {self.temporal_emb_type}...")

                tokenizer = AutoTokenizer.from_pretrained(model_name)

                with torch.no_grad():
                    prompt_inputs = tokenizer(
                        self.prompt,
                        return_tensors="pt"
                    ).to(self.device)

                    # ============================================================
                    # 同じ token がプロンプト中に複数回出現した場合、
                    # 最初の1個だけを Codebook に残す
                    # ============================================================
                    input_ids = prompt_inputs.input_ids[0]

                    unique_positions = []
                    seen_token_ids = set()

                    for pos, token_id in enumerate(input_ids.tolist()):
                        seen_token_ids.add(token_id)
                        unique_positions.append(pos)

                    unique_positions = torch.tensor(
                        unique_positions,
                        device=self.device,
                        dtype=torch.long
                    )
                    # 重複除去後の Codebook と対応する token ID を保存
                    self.register_buffer(
                        'codebook_token_ids',
                        input_ids[unique_positions]
                    )

                    print(
                        f"Original prompt tokens : {len(input_ids)}"
                    )
                    print(
                        f"Unique prompt tokens   : {len(unique_positions)}"
                    )

                    # デバッグ用：残した token を表示
                    unique_tokens = tokenizer.convert_ids_to_tokens(
                        input_ids[unique_positions].tolist()
                    )

                    print(
                        f"Unique Codebook Tokens : {unique_tokens}"
                    )
                    
                    if self.temporal_emb_type in [
                        'MLP_token_renew_loss_ph',
                        'MLP_token_renew_topk_loss_ph',
                        'MLP_token_renew_loss_ph_label'
                    ]:
                        self.codebook_words = unique_tokens

                        # ----------------------------
                        # Input Embedding
                        # ----------------------------
                        prompt_embs = self.llm.get_input_embeddings()(
                            prompt_inputs.input_ids
                        )[0]

                        # 重複 token を除去
                        prompt_embs = prompt_embs[unique_positions]

                        self.register_buffer(
                            'codebook_p',
                            prompt_embs
                        )

                        # ----------------------------
                        # Hidden Embedding
                        # ----------------------------
                        outputs = self.llm(**prompt_inputs)

                        prompt_hidden = outputs.last_hidden_state[0]

                        # P と完全に同じ token position を使用
                        prompt_hidden = prompt_hidden[unique_positions]

                        self.register_buffer(
                            'codebook_h',
                            prompt_hidden
                        )

                        print(
                            f"Prompt Embedding Codebook(P) shape: "
                            f"{self.codebook_p.shape}"
                        )

                        print(
                            f"Prompt Hidden Codebook(H) shape: "
                            f"{self.codebook_h.shape}"
                        )

        elif self.temporal_emb_type=='float':
            pass
        
        else:
            raise KeyError(f'Temporal embedding type {self.temporal_emb_type} not implemented.')


    def embed_event_text(self, event_texts: List[str], add_special_tokens: bool = False) -> List[Tensor]:
        """
        Embed the texts of events

        :param event_texts: event texts
        :param add_special_tokens: add special tokens or not
        :return: token embeddings of event texts, [(text_token_len, embedding_dim), ...]
        """
        event_tokens = self.tokenizer(
            event_texts, return_tensors='pt', add_special_tokens=add_special_tokens,
            padding=True, truncation=False)
        
        nums_tokens = event_tokens['attention_mask'].to(self.device).sum(dim=-1)
        event_embeddings_padded = self.llm_embedder(event_tokens['input_ids'].to(self.device))
        event_embeddings = [
            event_embedding_padded[:num_tokens]
            for num_tokens, event_embedding_padded in zip(nums_tokens, event_embeddings_padded)]
        
        return event_embeddings
    
    def forward(
        self, batch_event_times: List[Tensor], 
        batch_event_texts: List[List[str]],batch_event_lat:List[Tensor],batch_event_lon:List[Tensor],batch_event_dis:List[Tensor],batch_event_types: List[Tensor] = None) -> Tuple[List[Tensor]]:
        """
        Forward function to get hidden states of event sequences in a batch

        :param batch_event_times: batch of event times in sequences, [(seq_len,), ...]
        :param batch_event_time_deltas: batch of event time deltas in sequences, [(seq_len,), ...]
        :param batch_event_texts: batch of event texts in sequences, [(seq_len,), ...]
        :return: hidden states of events, [(seq_len, hidden_size), ...]
        """
        batch_sequence_embeddings = []
        batch_attention_masks = []
        batch_event_emb_indices = []
        batch_event_emb_indices_hidden = []

        # Lists to save data for loss computation

        self.temp_embs_for_loss = []
        

        # Process each event sequence in the batch
        for event_times, event_texts,event_lat,event_lon,event_dis ,event_types in zip(batch_event_times,
                                                               batch_event_texts,batch_event_lat,batch_event_lon,batch_event_dis, batch_event_types):
            
            # model.py の 324行目付近
            # =============== 変更: 数値データの標準化 (Standardization) ===============
            # ゼロ除算を回避するために 1e-8 を加算
            if self.temporal_emb_type in ['float','main']:
                pass
            else:
                norm_event_times = (event_times - self.mean_time) / (self.std_time + 1e-8)
                norm_event_lat = (event_lat - self.mean_lat) / (self.std_lat + 1e-8)
                norm_event_lon = (event_lon - self.mean_lon) / (self.std_lon + 1e-8)
                norm_event_dis = (event_dis - self.mean_dis) / (self.std_dis + 1e-8)
            # =========================================================================

            prompt_token_embeddings_tensor = self.embed_event_text(event_texts=[self.prompt], add_special_tokens=True)[0]
            
            sequence_embeddings = []
            sequence_attention_mask = []
            event_emb_indices = []
            event_emb_indices_hidden = []

            if self.use_tokens:
                if self.use_natural_separators:
                    time_event_texts=["Time"]
                    type_event_texts=["Type"]
                    lat_event_texts=["Latitude"]
                    lon_event_texts=["Longitude"]
                    dis_event_texts=["Distance"]
                    fusion_event_texts=["State"]
                    
                    
                else:
                    time_event_texts=["<|time_prefix|>"]
                    type_event_texts=["<|type_prefix|>"]
                    lat_event_texts=["<|latitude_prefix|>"]
                    lon_event_texts=["<|longitude_prefix|>"]
                    dis_event_texts=["<|distance_prefix|>"]
            
            
            
            # Add the prompt embeddings
            for prompt_token_embedding in prompt_token_embeddings_tensor :
                sequence_embeddings.append(prompt_token_embedding)
                sequence_attention_mask.append(1)

            # =============== 変更: 埋め込み層への入力 (正規化された値を使用) ===============
            # 時間の埋め込み (TPEの場合は絶対時間を使い、MLP/Linearの場合は正規化値を使用)

            if self.temporal_emb_type=='float':
                temporal_embeddings=self.embed_event_text(event_times)
                event_lat_embeddings=self.embed_event_text(event_lat)
                event_lon_embeddings=self.embed_event_text(event_lon)
                event_dis_embeddings=self.embed_event_text(event_dis)

            
            elif self.temporal_emb_type in ['MLP_token_renew_loss_ph_label']:
                time_scalar = norm_event_times.unsqueeze(-1)
                lat_scalar  = norm_event_lat.unsqueeze(-1)
                lon_scalar  = norm_event_lon.unsqueeze(-1)
                dis_scalar  = norm_event_dis.unsqueeze(-1)
                fusion_cat = torch.cat([time_scalar, lat_scalar, lon_scalar,dis_scalar,], dim=-1)
                fusion_vecs = self.fusion_mlp(fusion_cat)

                self.temp_embs_for_loss.append(fusion_vecs)

            elif self.temporal_emb_type in ['MLP_token_renew_vocab_topk_loss_ph','MLP_token_renew_vocab_loss_ph','MLP_token_renew_loss', 'MLP_token_renew_loss_p', 'MLP_token_renew_loss_h', 'MLP_token_renew_loss_ph','MLP_token_renew_loss_ph','MLP_token_renew_topk_loss_p', 'MLP_token_renew_topk_loss_ph']:
                time_scalar = norm_event_times.unsqueeze(-1)
                lat_scalar  = norm_event_lat.unsqueeze(-1)
                lon_scalar  = norm_event_lon.unsqueeze(-1)
                dis_scalar  = norm_event_dis.unsqueeze(-1)
                type_one_hot = F.one_hot(event_types, num_classes=self.num_event_types).to(self.dtype)
                # 全部まとめて結合 (Concatenate)
                fusion_cat = torch.cat([time_scalar, lat_scalar, lon_scalar,dis_scalar, type_one_hot], dim=-1)
                # MLPの出力を一度別の変数で受ける
                fusion_vecs = self.fusion_mlp(fusion_cat) # (seq_len, embedding_dim)

                self.temp_embs_for_loss.append(fusion_vecs)
            elif self.temporal_emb_type =='main':
                temporal_embeddings = self.time_mlp(event_times.unsqueeze(-1))
            else:
                temporal_embeddings = self.time_mlp(norm_event_times.unsqueeze(-1))
                event_lat_embeddings = self.lat_mlp(norm_event_lat.unsqueeze(-1))
                event_lon_embeddings = self.lon_mlp(norm_event_lon.unsqueeze(-1))
                event_dis_embeddings = self.dis_mlp(norm_event_dis.unsqueeze(-1))
            # =========================================================================
            if self.use_tokens:
                
                time_event_text_embedding= self.embed_event_text(time_event_texts)
                type_event_text_embedding= self.embed_event_text(type_event_texts)
                lat_event_text_embedding= self.embed_event_text(lat_event_texts)
                lon_event_text_embedding= self.embed_event_text(lon_event_texts)
                dis_event_text_embedding= self.embed_event_text(dis_event_texts)
                if self.use_natural_separators:
                    fusion_event_text_embedding= self.embed_event_text(fusion_event_texts)
            
            event_text_embeddings = self.embed_event_text(event_texts)  # [(text_token_len, embedding_dim), ...]
            
            if self.temporal_emb_type in ['MLP_token_renew_vocab_topk_loss_ph','MLP_token_renew_loss_ph', 'MLP_token_renew_topk_loss_ph']:
                    for fusion_vec in fusion_vecs:
                    # Add the event time embedding, temporal_embedding: (embedding_dim,)
                        sequence_embeddings.append(fusion_vec)
                        sequence_attention_mask.append(1)
                        event_emb_indices.append(len(sequence_embeddings) - 1)

            elif self.temporal_emb_type in ['MLP_token_renew_loss_ph_label']:
                for fusion_vec,event_token_embedding in zip(fusion_vecs,event_text_embeddings):
                        # Add the event time embedding, temporal_embedding: (embedding_dim,)
                    if self.use_tokens:
                        sequence_embeddings.append(fusion_event_text_embedding[0].squeeze(0))
                        sequence_attention_mask.append(1)
                    sequence_embeddings.append(fusion_vec)
                    sequence_attention_mask.append(1)
                    event_emb_indices_hidden.append(len(sequence_embeddings) - 1)
                    # Add event text token embeddings, event_token_embedding: (embedding_dim,)
                    if self.use_tokens:
                        sequence_embeddings.append(type_event_text_embedding[0].squeeze(0))
                        sequence_attention_mask.append(1)
                    for event_token_embedding1 in event_token_embedding:
                        sequence_embeddings.append(event_token_embedding1)
                        sequence_attention_mask.append(1)
                    
                    event_emb_indices.append(len(sequence_embeddings) - 1)

            elif self.temporal_emb_type=='main':

                for temporal_embedding, event_token_embedding in zip(temporal_embeddings,event_text_embeddings):
                    # Add the event time embedding, temporal_embedding: (embedding_dim,)
                    sequence_embeddings.append(temporal_embedding)
                    sequence_attention_mask.append(1)
                    for event_token_embedding1 in event_token_embedding:
                        sequence_embeddings.append(event_token_embedding1)
                        sequence_attention_mask.append(1)
                    event_emb_indices.append(len(sequence_embeddings) - 1)
            else:
            
                for temporal_embedding, event_token_embedding,event_lat_embedding,event_lon_embedding,event_dis_embedding in zip(temporal_embeddings,event_text_embeddings,event_lat_embeddings,event_lon_embeddings,event_dis_embeddings):
                    
                    
                    
                    # Add the event time embedding, temporal_embedding: (embedding_dim,)
                    if self.use_tokens:
                        sequence_embeddings.append(time_event_text_embedding[0].squeeze(0))
                        sequence_attention_mask.append(1)
                    
                    if self.temporal_emb_type=='float':
                        for temporal_embedding2 in temporal_embedding:
                            sequence_embeddings.append(temporal_embedding2)
                            sequence_attention_mask.append(1)
                    else:
                        sequence_embeddings.append(temporal_embedding)
                        sequence_attention_mask.append(1)

                    # Add event text token embeddings, event_token_embedding: (embedding_dim,)
                    if self.use_tokens:
                        sequence_embeddings.append(type_event_text_embedding[0].squeeze(0))
                        sequence_attention_mask.append(1)
                    for event_token_embedding1 in event_token_embedding:
                        sequence_embeddings.append(event_token_embedding1)
                        sequence_attention_mask.append(1)

                    if self.use_tokens:
                        for token_emb in lat_event_text_embedding[0]:
                            sequence_embeddings.append(token_emb)
                            sequence_attention_mask.append(1)
                    if self.temporal_emb_type=='float':
                        for event_lat_embedding1 in event_lat_embedding:
                            sequence_embeddings.append(event_lat_embedding1)
                            sequence_attention_mask.append(1)
                    else:
                        sequence_embeddings.append(event_lat_embedding)
                        sequence_attention_mask.append(1)

                    if self.use_tokens:
                        for token_emb in lon_event_text_embedding[0]:
                            sequence_embeddings.append(token_emb)
                            sequence_attention_mask.append(1)
                    if self.temporal_emb_type=='float':
                        for event_lon_embedding1 in event_lon_embedding:
                            sequence_embeddings.append(event_lon_embedding1)
                            sequence_attention_mask.append(1)
                    else:
                        sequence_embeddings.append(event_lon_embedding)
                        sequence_attention_mask.append(1)

                    if self.use_tokens:
                        for token_emb in dis_event_text_embedding[0]:
                            sequence_embeddings.append(token_emb)
                            sequence_attention_mask.append(1)
                    if self.temporal_emb_type=='float':
                        for event_dis_embedding1 in event_dis_embedding:
                            sequence_embeddings.append(event_dis_embedding1)
                            sequence_attention_mask.append(1)
                    else:
                        sequence_embeddings.append(event_dis_embedding)
                        sequence_attention_mask.append(1)

                    event_emb_indices.append(len(sequence_embeddings) - 1)

            # Convert sequence embeddings to tensors
            batch_sequence_embeddings.append(torch.stack(sequence_embeddings))  # [(seq_emb_len, embedding_dim), ...]
            batch_attention_masks.append(torch.tensor(sequence_attention_mask).to(self.device))  # [(seq_emb_len,), ...]
            batch_event_emb_indices.append(torch.tensor(event_emb_indices).to(self.device))  # [(seq_len,), ...]
            batch_event_emb_indices_hidden.append(torch.tensor(event_emb_indices_hidden).to(self.device))

        # Pad the sequence embeddings in the batch
        padded_embeddings = torch.nn.utils.rnn.pad_sequence(
            batch_sequence_embeddings, batch_first=True)  # (num_seqs, max_seq_emb_len, embedding_dim)
        padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
            batch_attention_masks, batch_first=True)  # (num_seqs, max_seq_emb_len)
        
        # Pass the padded embeddings through the LLM
        llm_output = self.llm(
            inputs_embeds=padded_embeddings, attention_mask=padded_attention_masks,
        ).last_hidden_state  # (batch_size, max_seq_emb_len, hidden_size)

        # Collect the hidden states at the last event embedding positions
        batch_hidden_states = [
            llm_output[i, batch_event_emb_indices[i], :]
            for i in range(len(batch_event_emb_indices))
        ]  # [(seq_len, hidden_size), ...]

        if self.temporal_emb_type in ['MLP_token_renew_loss_ph_label']:
            batch_hidden_states_hidden = [
                llm_output[i, batch_event_emb_indices_hidden[i], :]
                for i in range(len(batch_event_emb_indices_hidden))
            ]  
            return batch_hidden_states, batch_hidden_states_hidden
        return batch_hidden_states

    def compute_intensities(
        self, batch_event_time_deltas: List[Tensor], batch_hidden_states: List[Tensor],
    ) -> List[Tensor]:
        """
        Compute intensities from event times and hidden states

        :param batch_event_time_deltas: event time deltas in a batch of event sequences, [(seq_len, n_samples), ...]
        :param batch_hidden_states: hidden states for a batch of event sequences, [(seq_len, hidden_size), ...]
        :return: a batch of event intensities, [(seq_len - 1, num_types), ...]
        """
        batch_intensities = []

        for event_time_deltas, hidden_states in zip(batch_event_time_deltas, batch_hidden_states):
            event_time_deltas_tensor = event_time_deltas.unsqueeze(1)[1:]  # (seq_len - 1, 1)
            intensities_current = self.intensity_current(event_time_deltas_tensor)  # (seq_len - 1, num_types)
            intensities_history = self.intensity_history(hidden_states[:-1])  # (seq_len - 1, num_types)
            intensities = self.softplus(intensities_current + intensities_history)  # (seq_len - 1, num_types)
            batch_intensities.append(intensities)  # [(seq_len - 1, num_types), ...]

        return batch_intensities

    def generate_time_deltas(self, batch_event_time_deltas: List[Tensor]) -> List[Tensor]:
        """
        Generate the time delta samples for every interval without padding.

        :param batch_event_time_deltas: list of event times since the last event, [(seq_len,), ...]
        :return: list of time samples for every interval, [(seq_len - 1, n_samples), ...]
        """
        batch_sampled_time_deltas = []

        # Process each sequence in the batch
        for event_time_deltas in batch_event_time_deltas:
            # Convert the tensor of time deltas
            event_time_deltas_tensor = event_time_deltas.unsqueeze(1)[1:]  # (seq_len - 1, 1)

            # Generate the sampling ratios
            time_delta_ratios = torch.linspace(
                start=0.0, end=1.0, steps=self.num_integral_samples, device=self.device)  # (1, n_samples)

            # Sample the time deltas across the intervals
            sampled_time_deltas = event_time_deltas_tensor * time_delta_ratios.unsqueeze(0)  # (seq_len - 1, n_samples)
            batch_sampled_time_deltas.append(sampled_time_deltas)  # [(seq_len - 1, n_samples), ...]

        return batch_sampled_time_deltas

    def compute_sampled_intensities(
        self, batch_sampled_time_deltas: List[Tensor], batch_hidden_states: List[Tensor],
    ) -> List[Tensor]:
        """
        Compute intensities at sampled time deltas in a batch

        :param batch_sampled_time_deltas: a batch of sampled time delta sequence, [(seq_len - 1, n_samples), ...]
        :param batch_hidden_states: a batch of hidden state sequences, [(seq_len, hidden_size), ...]
        :return: a batch of intensities at sampled times, [(seq_len - 1, n_samples, num_types), ...]
        """
        batch_sampled_intensities = []

        for sampled_time_deltas, hidden_states in zip(batch_sampled_time_deltas, batch_hidden_states):
            sampled_time_deltas_tensor = sampled_time_deltas.unsqueeze(-1)  # (seq_len - 1, n_samples, 1)
            sampled_intensities_current = self.intensity_current(
                sampled_time_deltas_tensor)  # (seq_len - 1, n_samples, num_types)
            sampled_intensities_history = self.intensity_history(
                hidden_states[:-1]).unsqueeze(1)  # (seq_len - 1, 1, num_types)
            sampled_intensities = self.softplus(
                sampled_intensities_current + sampled_intensities_history)  # (seq_len - 1, n_samples, num_types)
            batch_sampled_intensities.append(sampled_intensities)  # [(seq_len - 1, n_samples, num_types), ...]

        return batch_sampled_intensities

    def compute_log_likelihood(
        self, batch_event_time_deltas: List[Tensor], batch_event_types: List[Tensor],
        batch_hidden_states: List[Tensor]) -> List[Tensor]:
        """
        Compute log likelihoods of event sequences in a batch

        :param batch_event_time_deltas: a batch of event time deltas, [(seq_len,), ...]
        :param batch_event_types: a batch of event types, [(seq_len,), ...]
        :param batch_hidden_states: a batch of hidden states, [(seq_len, hidden_size), ...]
        :return: a batch of log likelihoods, [(seq_len - 1,), ...]
        """
        # Compute intensities of events
        batch_event_intensities = self.compute_intensities(
            batch_event_time_deltas=batch_event_time_deltas,
            batch_hidden_states=batch_hidden_states)  # [(seq_len - 1, num_types), ...]

        # Sample time deltas and compute their intensities
        batch_sampled_time_deltas = self.generate_time_deltas(
            batch_event_time_deltas=batch_event_time_deltas)  # [(seq_len - 1, n_samples), ...]
        batch_sampled_intensities = self.compute_sampled_intensities(
            batch_sampled_time_deltas=batch_sampled_time_deltas,
            batch_hidden_states=batch_hidden_states)  # [(seq_len - 1, n_samples, num_types), ...]

        # Compute log likelihoods for the event part
        batch_log_likelihoods = []

        # event_types: (seq_len,), event_time_deltas: (seq_len,)
        # event_intensities: (seq_len - 1, num_types), sampled_intensities: (seq_len - 1, n_samples, num_types)
        for event_types, event_time_deltas, event_intensities, sampled_intensities in zip(
            batch_event_types, batch_event_time_deltas, batch_event_intensities, batch_sampled_intensities):
            # Compute the log likelihood part of events
            event_type_masks = F.one_hot(
                event_types[1:], num_classes=self.num_event_types).to(self.device)  # (seq_len - 1, num_types)
            event_likelihoods = torch.sum(event_intensities * event_type_masks, dim=-1)  # (seq_len - 1,)
            event_log_likelihoods = torch.log(event_likelihoods + 1e-8)  # (seq_len - 1,)

            # Compute the log likelihood part of non-events
            sampled_total_intensities = torch.sum(sampled_intensities, dim=-1)  # (seq_len - 1, n_samples)
            non_event_log_likelihoods = \
                sampled_total_intensities.mean(dim=-1) * event_time_deltas[1:]  # (seq_len - 1,)

            log_likelihoods = event_log_likelihoods - non_event_log_likelihoods  # (seq_len - 1,)
            batch_log_likelihoods.append(log_likelihoods)

        return batch_log_likelihoods

    def compute_loss(self, batch: Dict[str, list]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute the loss terms

        :param batch: a batch of event sequences
        :return: numbers of events, negative log likelihood (NLL) losses, event type prediction losses,
            event time prediction losses
        """
        
        batch_event_times = batch['time_since_start']
        batch_event_time_deltas = batch['time_since_last_event']
        batch_event_types = batch['type_event']
        batch_event_texts=batch['type_text']
        
        batch_event_lat = batch['latitude']
        batch_event_lon = batch['longitude']
        batch_event_dis = batch['distance']
        batch_event_time_float=batch['time_float']
        batch_event_lat_float=batch['latitude_float']
        batch_event_lon_float=batch['longitude_float']
        batch_event_dis_float=batch['distance_float']

        if self.temporal_emb_type=='float':
            # Compute the hidden states
            batch_hidden_states = self.forward(
                batch_event_times=batch_event_time_float,
                batch_event_texts=batch_event_texts,
                batch_event_lat=batch_event_lat_float,
                batch_event_lon=batch_event_lon_float,
                batch_event_dis=batch_event_dis_float,
                batch_event_types=batch_event_types)  # [(seq_len, hidden_size), ...]
        elif self.temporal_emb_type == 'MLP_token_renew_loss_ph_label':
            batch_hidden_states ,batch_hidden_states_hidden = self.forward(
                batch_event_times=batch_event_time_float,
                batch_event_texts=batch_event_texts,
                batch_event_lat=batch_event_lat_float,
                batch_event_lon=batch_event_lon_float,
                batch_event_dis=batch_event_dis_float,
                batch_event_types=batch_event_types)  # [(seq_len, hidden_size), ...]
        
        else:
        # Compute the hidden states
            batch_hidden_states = self.forward(
                batch_event_times=batch_event_times,
                batch_event_texts=batch_event_texts,
                batch_event_lat=batch_event_lat,
                batch_event_lon=batch_event_lon,
                batch_event_dis= batch_event_dis,
                batch_event_types=batch_event_types)  # [(seq_len, hidden_size), ...]

         # ============================================================
        # t-SNE / raw_data 保存用
        # 学習に使っているTensorそのものを保持せず、
        # detachしたCPU Tensorとしてコピーを作る
        # ============================================================
        # runner.py のバッチ内ループ（raw_data 保存部分）
        if not self.training:  # 👈 評価（eval）の時だけ詳細データを保存する！
            saved_input_vectors = []

            if hasattr(
                self,
                'temp_embs_for_loss'
            ):
                saved_input_vectors = [
                    emb.detach().cpu()
                    for emb in self.temp_embs_for_loss
                ]

            saved_hidden_vectors = [
                hidden.detach().cpu()
                for hidden in batch_hidden_states
            ]
        else:
            saved_input_vectors = []
            saved_hidden_vectors = []

        
        # Compute the log likelihoods
        batch_log_likelihoods = self.compute_log_likelihood(
            batch_event_time_deltas=batch_event_time_deltas,
            batch_event_types=batch_event_types,
            batch_hidden_states=batch_hidden_states)  # [(seq_len - 1,), ...]

        # Predict the next events
        batch_next_event_type_probs, batch_next_event_times = self.predict_next_event_probs(
            batch_hidden_states=batch_hidden_states)

        batch_nll_losses = []
        batch_type_losses = []
        batch_time_losses = []

        for log_likelihoods, next_event_type_probs, event_types, next_event_times, event_times in zip(
            batch_log_likelihoods, batch_next_event_type_probs, batch_event_types, batch_next_event_times,
            batch_event_times):
            nll_loss = - torch.sum(log_likelihoods, dim=0)
            event_type_masks = F.one_hot(
                event_types[1:], num_classes=self.num_event_types).to(self.device)  # (seq_len - 1, num_types)

            type_loss = torch.sum(- event_type_masks * torch.log(next_event_type_probs[:-1])+ 1e-8)
            time_loss = torch.sum((next_event_times[:-1] - event_times[1:]) ** 2, dim=0)

            batch_nll_losses.append(nll_loss)
            batch_type_losses.append(type_loss)
            batch_time_losses.append(time_loss)

        batch_event_nums = torch.LongTensor([len(event_times) - 1 for event_times in batch_event_times])
        batch_nll_losses = torch.stack(batch_nll_losses)
        batch_type_losses = torch.stack(batch_type_losses)
        batch_time_losses = torch.stack(batch_time_losses)

# ------------------ デバッグ出力用ヘルパー関数 ------------------
        def debug_print_topk(dist_sq, indices, name_label="", query_vec=None):
            if self.training:
                return
            if not hasattr(self, f'_printed_topk_{name_label}'):
                sample_indices = indices[0].tolist()
                sample_dist_sq = dist_sq[0].tolist()
                
                log_lines = []
                log_lines.append(f"\n🌟 [Debug: {name_label}] 最初のイベントが引き寄せられた Top-{len(sample_indices)} の単語:")

# 🌟 追加: ベクトルが渡されている場合は L2 Norm (原点からの距離) を計算して表示
                if query_vec is not None:
                    vec_norm = torch.norm(query_vec[0], p=2).item()
                    log_lines.append(f"   📌 [L2 Norm] 最初のイベントベクトルの原点からの距離: {vec_norm:.4f}")
                
                try:
                    # 🌟 500単語辞書の場合はリストから直接引く
                    if hasattr(self, 'codebook_words'):
                        selected_tokens = [self.codebook_words[idx] for idx in sample_indices]
                    else:
                        prompt_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids[0]
                        selected_tokens = self.tokenizer.convert_ids_to_tokens(prompt_ids[sample_indices])
                        
                    for rank, (tok, d2, idx) in enumerate(zip(selected_tokens, sample_dist_sq, sample_indices)):
                        d = d2 ** 0.5
                        log_lines.append(f"  {rank+1}位: '{tok}' (位置:{idx}) | 距離: {d:.4f} (距離の2乗: {d2:.4f})")
                except Exception as e:
                    for rank, (d2, idx) in enumerate(zip(sample_dist_sq, sample_indices)):
                        d = d2 ** 0.5
                        log_lines.append(f"  {rank+1}位: インデックス {idx} | 距離: {d:.4f} (距離の2乗: {d2:.4f})")
                
                log_text = "\n".join(log_lines)
                print(log_text)
                
                try:
                    save_dir = getattr(self, 'result_save_path', '.')
                    log_file_path = f"{save_dir}/topk_debug_log.txt"
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(log_text + "\n")
                except Exception as e:
                    pass
                    
                setattr(self, f'_printed_topk_{name_label}', True)
        # ----------------------------------------------------------------

        batch_words_p = []
        batch_words_h = []
        batch_distination_p=[]
        batch_distination_h=[]
        batch_dist_eq_p = []  # earthquakeとの距離 (Input)
        batch_dist_eq_h = []  # earthquakeとの距離 (Hidden)
        
        
        
        if self.temporal_emb_type in ['MLP_token_renew_loss_ph_label','MLP_token_renew_vocab_topk_loss_ph','MLP_token_renew_vocab_loss_ph_scaled','MLP_token_renew_loss_ph', 'MLP_token_renew_topk_loss_ph', 'MLP_token_renew_vocab_loss_ph'] and hasattr(self, 'codebook_p') and hasattr(self, 'codebook_h'):
            k_val_p = min(5, self.codebook_p.shape[0])
            k_val_h = min(5, self.codebook_h.shape[0])
                    
            # ① 入力埋め込み (P) に対する制約計算
            seq_losses_p = []
            seq_eq_p_list = []  
            for seq_emb in self.temp_embs_for_loss:
                        
                dist_p = torch.cdist(seq_emb, self.codebook_p, p=2)
        
                # Eval(評価)時のみ、最短距離の単語をデコードして保存
                if not self.training:
                    min_dist_p, min_indices_p = torch.min(dist_p, dim=-1)
                    dist_to_eq_p = dist_p[:, 0]  #  0番目の単語 (earthquake) との距離
        
                    words_p = [self.codebook_words[idx] for idx in min_indices_p.cpu().tolist()]
                    batch_words_p.append(words_p)
                                
                    batch_distination_p.append(min_dist_p.tolist())
                    seq_eq_p_list.append(dist_to_eq_p.tolist())  # 🌟 追加
        
                if self.temporal_emb_type in ['MLP_token_renew_topk_loss_ph','MLP_token_renew_vocab_topk_loss_ph']:
                    topk_dist_sq_p, topk_indices_p = torch.topk(dist_p**2, k=k_val_p, largest=False, dim=-1)
                    seq_loss_p = (torch.sum(topk_dist_sq_p, dim=-1) / k_val_p).mean()
                    debug_print_topk(topk_dist_sq_p, topk_indices_p, "Codebook_P", query_vec=seq_emb)
                else:
                    min_dist_sq, min_indices = torch.min(dist_p**2, dim=-1)
                    seq_loss_p = min_dist_sq.mean()
                    debug_print_topk(min_dist_sq.unsqueeze(-1), min_indices.unsqueeze(-1), "Codebook_P_HardMin", query_vec=seq_emb) # 👈 追加
                seq_losses_p.append(seq_loss_p)
            cb_losses_p = torch.stack(seq_losses_p).to(batch_nll_losses.device)
            if not self.training and seq_eq_p_list:
                batch_dist_eq_p.extend(seq_eq_p_list)  
            self.temp_embs_for_loss = []
        
        # ② 隠れ層 (H) に対する制約計算
            seq_losses_h = []
            seq_eq_h_list = []  
            if self.temporal_emb_type == 'MLP_token_renew_loss_ph_label':
                target_hidden_states = batch_hidden_states_hidden
            else:
                target_hidden_states = batch_hidden_states
            for h_state in target_hidden_states: 
                dist_h = torch.cdist(h_state, self.codebook_h, p=2)
        
        #  Eval(評価)時のみ、最短距離の単語をデコードして保存
                if not self.training:
                    min_dist_h, min_indices_h = torch.min(dist_h, dim=-1)
                    dist_to_eq_h = dist_h[:, 0]  # 🌟 0番目の単語 (earthquake) との距離

                    words_h = [self.codebook_words[idx] for idx in min_indices_h.cpu().tolist()]
                    batch_words_h.append(words_h)
                    
                    batch_distination_h.append(min_dist_h.tolist())
                    seq_eq_h_list.append(dist_to_eq_h.tolist())  
        
                if self.temporal_emb_type in ['MLP_token_renew_topk_loss_ph','MLP_token_renew_vocab_topk_loss_ph']:
                    topk_dist_sq_h, topk_indices_h = torch.topk(dist_h**2, k=k_val_h, largest=False, dim=-1)
                    seq_loss_h = (torch.sum(topk_dist_sq_h, dim=-1) / k_val_h).mean()
                    debug_print_topk(topk_dist_sq_h, topk_indices_h, "Codebook_H", query_vec=h_state)
                else:
                    min_dist_sq, min_indices = torch.min(dist_h**2, dim=-1)
                    seq_loss_h = min_dist_sq.mean()
                    debug_print_topk(min_dist_sq.unsqueeze(-1), min_indices.unsqueeze(-1), "Codebook_H_HardMin", query_vec=h_state) # 👈 追加
                                                                
                seq_losses_h.append(seq_loss_h)
            cb_losses_h = torch.stack(seq_losses_h).to(batch_nll_losses.device)
            if not self.training and seq_eq_h_list:
                batch_dist_eq_h.extend(seq_eq_h_list)  
            self.temp_embs_for_loss = []
                
            # ③ 両方の制約Lossを足し合わせる
            codebook_losses = cb_losses_p *self.codebook_loss_weight_p+cb_losses_h *self.codebook_loss_weight_h
                    
        else:
            codebook_losses = None
            cb_losses_p = None  
            cb_losses_h = None  

        batch_next_event_types = [
            next_event_type_probs.argmax(dim=-1) for next_event_type_probs in batch_next_event_type_probs]

        return batch_event_nums, batch_nll_losses, batch_type_losses, batch_time_losses, batch_next_event_types, batch_next_event_times,codebook_losses, batch_words_p, batch_words_h,batch_distination_p,batch_distination_h,batch_dist_eq_p, batch_dist_eq_h, cb_losses_p, cb_losses_h,saved_input_vectors,saved_hidden_vectors
            
    def predict_next_event_probs(self, batch_hidden_states: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Predict next events with probabilities (for training)

        :param batch_hidden_states: hidden states for a batch of event sequences, [(seq_len, hidden_size), ...]
        :return: a batch of next event type probabilities (in [(seq_len, num_types), ...])
            and next event times (in [(seq_len,), ...])
        """
        batch_next_event_type_probs = []
        batch_next_event_times = []

        for hidden_states in batch_hidden_states:
            next_event_type_probs = self.head_type(hidden_states)  # (seq_len, num_types)
            next_event_times = self.head_time(hidden_states).squeeze()  # (seq_len,)
            batch_next_event_type_probs.append(next_event_type_probs)  # [(seq_len, num_types), ...]
            batch_next_event_times.append(next_event_times)  # [(seq_len,), ...]

        return batch_next_event_type_probs, batch_next_event_times

    def save_models(self, model_weight_path: str):
        """
        Save the LoRA adapter and additional token weights without affecting training continuation.
        """
        print(f"Saving adapter and tokens to {model_weight_path}...")

        os.makedirs(model_weight_path, exist_ok=True)
    
        # 1. Save LoRA adapter (without merging)
        self.llm.save_pretrained(model_weight_path + '/model')

        # 2. Save new token weights separately
        if hasattr(self, 'new_token_weights') and self.new_token_weights is not None:
            torch.save(self.new_token_weights, os.path.join(model_weight_path, 'new_token_weights.pt'))
            print("New token weights saved separately.")

        # 3. Save tokenizer
        self.tokenizer.save_pretrained(model_weight_path + '/tokenizer')

        print("Save complete (Training can be continued).")

    def reset_topk_debug_flags(self):
        """エポックごとにTop-kデバッグログを再出力するため、フラグをリセットする"""
        keys = list(self.__dict__.keys())
        for attr in keys:
            if attr.startswith('_printed_topk_'):
                delattr(self, attr) # フラグを削除（次回また出力されるようになる）