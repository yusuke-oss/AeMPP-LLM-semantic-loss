"""
Train a TPP-LLM Model
"""
import argparse
import os
import sys
import json
import time
import torch
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType
import traceback  # ⬅️ これを追加！

# Local Imports
# Note: get_all_data_stats should be moved to data.py
from src.tpp_llm.taxi_renew_vocab2.data import (
    TPPLLMDataset, collate_fn, get_all_data_stats,get_collate_fn
)
from src.tpp_llm.taxi_renew_vocab2.model import TPPLLMModel
from src.tpp_llm.taxi_renew_vocab2.runner import TPPLLMRunner
from src.tpp_llm.taxi_renew_vocab2.utils import get_prompt
from src.tpp_llm.taxi_renew_vocab2.common_utils import seed_everything, seed_worker
from src.tpp_llm.taxi_renew_vocab2.analysis import perform_quantitative_analysis

# Enable anomaly detection for debugging (Can be set to False for production)
torch.autograd.set_detect_anomaly(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train and test the TPP-LLM model with event sequences.')
    # Model & Data Configuration
    parser.add_argument(
        '--model_path', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='llm path')
    parser.add_argument(
        '--dataset_path', type=str, required=True, help='dataset path')
    parser.add_argument(
        '--num_event_types', type=int, required=True, help='number of event types')
    parser.add_argument(
        '--temporal_emb_type', type=str, default='positional', choices=['MLP_token_renew_loss_ph_label','MLP_token_renew_vocab_topk_loss_ph_list','MLP_token_renew_vocab_topk_loss_ph','positional', 'MLP', 'linear', 'MLP_re', 'mlp_one', 'MLP_add', 'MLP_mul', 'mlp_one_add', 'mlp_one_mul','float','MLP_fusion','main','MLP_fusion_renew','MLP_token_renew_loss','MLP_token_renew_loss_p','MLP_token_renew_loss_h','MLP_token_renew_loss_ph','MLP_token_renew_topk_loss_p', 'MLP_token_renew_topk_loss_h', 'MLP_token_renew_topk_loss_ph','MLP_token_renew_vocab_loss_ph'],
        help='temporal embedding type')
    parser.add_argument(
        '--temporal_emb_first', action='store_true', help='temporal embedding first or not')
    parser.add_argument(
        '--no_prompt', action='store_true', help='no prompt')
    parser.add_argument(
        '--num_integral_samples', type=int, default=20, help='number of samples during one integral step')
    # Quantization & PEFT Configuration
    parser.add_argument(
        '--quant_type', type=str, default=None, choices=['4bit', '8bit', None], help='quantization type')
    parser.add_argument(
        '--peft_type', type=str, default=None, choices=['lora', 'None'], help='peft type')
    parser.add_argument(
        '--lora_rank', type=int, default=16, help='lora rank')
    parser.add_argument(
        '--lora_modules', type=str, nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        help='lora target modules')
    # Training Hyperparameters
    parser.add_argument(
        '--train_batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument(
        '--eval_batch_size', type=int, default=16, help='batch size for evaluation')
    parser.add_argument(
        '--learning_rate', type=float, default=5e-4, help='larning rate')
    parser.add_argument(
        '--lr_scheduler_type', type=str, default='constant', help='learning rate scheduler type')
    parser.add_argument(
        '--num_epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument(
        '--warmup_ratio', type=float, default=0, help='warmup ratio')
    parser.add_argument(
        '--beta_type', type=float, default=1, help='loss coefficient of the event type prediction')
    parser.add_argument(
        '--beta_time', type=float, default=1, help='loss coefficient of the event time prediction')
    # Environment & Save Paths
    parser.add_argument(
        '--device', type=str, default='cpu', help='cpu or cuda device')
    parser.add_argument(
        '--seed', type=int, default=2024, help='seed for reproducibility')
    parser.add_argument(
        '--result_save_path', type=str, default='',help='save result')
    parser.add_argument(
        '--model_weight_path', type=str, default='',help='save model weight')
    parser.add_argument(
        '--weight_path', type=str,default='', help='save weight')
    # Execution Flags
    parser.add_argument(
        '--save_flag',  action='store_true', default=False, help='save weight and model')
    parser.add_argument(
        '--load_flag',  action='store_true', default=False, help='load weight and model')
    parser.add_argument(
        '--train_flag',  action='store_true', default=False, help='train weight and model')
    parser.add_argument(
        '--flozen_flag',  action='store_true', default=False, help='train weight and model')
    parser.add_argument(
        '--use_natural_separators', action='store_true', default=False, help='Use "Time", "Type" instead of tokens')
    parser.add_argument(
        '--use_tokens', action='store_true', default=False, help='Use "Time", "Type" instead of tokens')
    # train_tpp_llm.py の parser 設定に追加
    parser.add_argument(
        '--emb_flozen_flag', action='store_true', default=False, help='Freeze temporal/mag/dep embedding layers')
    parser.add_argument('--resume_flag', action='store_true', help='resume flag')

    parser.add_argument(
        '--codebook_loss_weight_p', type=float, default=0.01, help='Weight for codebook constraint loss')
        # 🌟 【追加】コマンドラインから Codebook の重みを指定できるようにする引数
    parser.add_argument(
        '--codebook_loss_weight_h', type=float, default=0.01, help='Weight for codebook constraint loss')
    parser.add_argument(
        '--scale_flag', action='store_true', default=False, help='scale')
        
    parser.add_argument(
        '--llama_flag', action='store_true', default=False, help='llama')
    parser.add_argument(
        '--gemma_flag', action='store_true', default=False, help='gemma')
    parser.add_argument(
        '--llama3_flag', action='store_true', default=False, help='llama3')
    parser.add_argument(
            '--llama3_3_flag', action='store_true', default=False, help='llama3-3')
    parser.add_argument(
        '--no_mlp_flag', action='store_true', default=False, help='llama')
    parser.add_argument(
        '--mlp_flag', action='store_true', default=False, help='gemma')
        
    
    start_time = time.time()
    # Parse arguments
    args = parser.parse_args()

    if args.resume_flag:
        args.train_flag = True
        args.save_flag = True

    if args.temporal_emb_type=='linear_flozen':
        emb_flozen_flag=True
    
    
    print(f"=== TPP-LLM Training Configuration ===")
    print(f"Args: {args}")
    print(f"LR Scheduler: {args.lr_scheduler_type}")
    print(f"Warmup Ratio: {args.warmup_ratio}")
    print(f"codebook_loss_weight_p: {args.codebook_loss_weight_p}")
    print(f"codebook_loss_weight_h: {args.codebook_loss_weight_h}")
    print(f"scale_flag: {args.scale_flag}")
    print(f"llama_flag: {args.llama_flag}")
    print(f"gemma_flag: {args.gemma_flag}")
    print(f"llama3_flag: {args.llama3_flag}")
    print(f"llama3_3_flag: {args.llama3_3_flag}")
    print(f"no_mlp_flag: {args.no_mlp_flag}")
    print(f"mlp_flag: {args.mlp_flag}")
    print(f"======================================")

    # Reproducibility & Directory Setup
    seed_everything(args.seed)
    os.makedirs(args.result_save_path, exist_ok=True)

    # Initialize log file
    val_log_path = os.path.join(args.result_save_path, 'val.txt')
    with open(val_log_path, 'a') as f:
        f.write(f"Seed: {args.seed}\n")
        f.write(f"codebook_loss_weight_p: {args.codebook_loss_weight_p}\n")
        f.write(f"codebook_loss_weight_h: {args.codebook_loss_weight_h}\n")
        f.write(f"scale_flag: {args.scale_flag}\n")
        f.write(f"llama_flag: {args.llama_flag}\n")
        f.write(f"gemma_flag: {args.gemma_flag}\n")
        f.write(f"llama3_flag: {args.llama3_flag}\n")
        f.write(f"llama3_3_flag: {args.llama3_3_flag}\n")
        f.write(f"no_mlp_flag: {args.no_mlp_flag}\n")
        f.write(f"mlp_flag: {args.mlp_flag}\n")
        #print(f"scale_flag: {args.scale_flag}")
        if args.temporal_emb_type=='MLP':
            f.write(f"Beta Semantic: {args.beta_semantic}\n")
    
    # Prompt Preparation
    base_dataset_name = os.path.basename(args.dataset_path).replace('_few_shot', '')
    prompt = get_prompt(
        dataset_name=base_dataset_name, 
        event_time_first=args.temporal_emb_first,
        temporal_emb_type=args.temporal_emb_type,
        use_natural_separators=args.use_natural_separators  # ← これを追加！
    )
    if args.no_prompt:
        prompt = ''
    

    with open(val_log_path, 'a') as f:
        f.write(f"Prompt: {prompt}\n")

    
    # Get the quantization config
    if args.quant_type == '4bit':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.quant_type == '8bit':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=False,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    # Get the PEFT config
    if args.peft_type == 'lora':
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=16,
            target_modules=args.lora_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
    else:
        peft_config = None

    
    # Data Preparation & Statistics
    print("Loading dataset statistics...")
    stats = get_all_data_stats(args.dataset_path)

    print("Initializing DataLoaders...")
    dataset_train = TPPLLMDataset(f'{args.dataset_path}/train.json')
    dataset_val = TPPLLMDataset(f'{args.dataset_path}/dev.json')
    dataset_test = TPPLLMDataset(f'{args.dataset_path}/test.json')

    g = torch.Generator()
    g.manual_seed(args.seed)
    
    #dataloader_train = DataLoader(
    #    dataset_train, batch_size=args.train_batch_size, shuffle=True, 
    #    generator=g, worker_init_fn=seed_worker, collate_fn=collate_fn
    #)
    #dataloader_val = DataLoader(dataset_val, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    #dataloader_test = DataLoader(dataset_test, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.train_batch_size, shuffle=True, 
        generator=g, worker_init_fn=seed_worker, collate_fn=get_collate_fn(stats)
    )
    dataloader_val = DataLoader(dataset_val, batch_size=args.eval_batch_size, shuffle=False, collate_fn=get_collate_fn(stats))
    dataloader_test = DataLoader(dataset_test, batch_size=args.eval_batch_size, shuffle=False, collate_fn=get_collate_fn(stats))
    
    
    # 5. Model Initialization
    print("Initializing TPP-LLM Model...")
    model = TPPLLMModel(
        model_name=args.model_path,
        num_event_types=args.num_event_types,
        num_integral_samples=args.num_integral_samples,
        temporal_emb_type=args.temporal_emb_type,
        temporal_emb_first=args.temporal_emb_first,
        prompt=prompt,
        bnb_config=bnb_config,
        peft_config=peft_config,
        device=args.device,
        model_weight_path=args.model_weight_path,
        save_flag=args.save_flag,
        load_flag=args.load_flag,
        train_flag=args.train_flag,
        resume_flag=args.resume_flag, # ⬅️ これを追加
        flozen_flag=args.flozen_flag,
        use_natural_separators=args.use_natural_separators,
        use_tokens=args.use_tokens,
        emb_flozen_flag=args.emb_flozen_flag,
        codebook_loss_weight_p=args.codebook_loss_weight_p,  # 🌟 【追加】渡す引数に追加
        codebook_loss_weight_h=args.codebook_loss_weight_h,  # 🌟 【追加】渡す引数に追加
        scale_flag=args.scale_flag,
        llama_flag=args.llama_flag,
        gemma_flag=args.gemma_flag,
        llama3_flag=args.llama3_flag,
        llama3_3_flag=args.llama3_3_flag,
        no_mlp_flag=args.no_mlp_flag,
        mlp_flag=args.mlp_flag
       )

    # =========================================================
    # 🌟🌟🌟 追加: Codebook用のプロンプトをログ(val.txt)と画面に出力 🌟🌟🌟
    # =========================================================
    if hasattr(model, 'codebook_prompt'):
        codebook_prompt_log = f"Codebook Prompt: {model.codebook_prompt}\n"
        print(codebook_prompt_log.strip())
        with open(val_log_path, 'a', encoding='utf-8') as f:
            f.write(codebook_prompt_log)
    # =========================================================
    
    # ---------------------------------------------------------
    # 🛠️ SANITY CHECK: 凍結状態とトークンの確認
    # ---------------------------------------------------------
    sanity_log = []
    sanity_log.append("\n" + "="*60)
    sanity_log.append(" 🛠️ SANITY CHECK: FROZEN STATUS & TOKENS")
    sanity_log.append("="*60)
    
    # 1. LLM本体の凍結確認（埋め込み層とLoRA層以外）
    llm_trainable = sum(p.numel() for n, p in model.llm.named_parameters() if p.requires_grad and "embed" not in n and "lora" not in n)
    llm_status = "❌ 全面学習オン (危険)" if llm_trainable > 0 else "✅ 凍結 (Frozen)"
    sanity_log.append(f"[1] LLM本体 (ベース重み) : {llm_status} (学習可能パラメータ数: {llm_trainable})")

    # [追加] LoRAレイヤーだけの学習状態を個別で確認
    lora_trainable = sum(p.numel() for n, p in model.llm.named_parameters() if p.requires_grad and "lora" in n)
    sanity_log.append(f"[1.5] LoRAレイヤー        : {'✅ 学習オン' if lora_trainable > 0 else '❌ 凍結 or なし'} (パラメータ数: {lora_trainable})")

    # 2. トークン埋め込み層の確認
    emb_trainable = sum(p.numel() for p in model.llm_embedder.parameters() if p.requires_grad)
    emb_status = "✅ 学習オン" if emb_trainable > 0 else "❌ 凍結"
    sanity_log.append(f"[2] トークン埋め込み層  : {emb_status} (学習可能パラメータ数: {emb_trainable})")

    # 3. 時間・属性の線形変換/MLP層の凍結確認
    # ⬇️ 変更: リスト形式にして新しく追加したタイプを全て網羅
    if args.temporal_emb_type in ['MLP', 'MLP_add', 'MLP_mul']:
        time_mlp_trainable = sum(p.numel() for p in model.lat_mlp.parameters() if p.requires_grad)
        linear_status = "✅ 凍結 (MLP_flozen)" if time_mlp_trainable == 0 else "✅ 学習オン (linear/MLP)"
        sanity_log.append(f"[3] MLP (time_mlp) : {linear_status} (学習可能パラメータ数: {time_mlp_trainable})")
    
    elif args.temporal_emb_type in ['MLP_re', 'mlp_re', 'mlp_one', 'mlp_one_add', 'mlp_one_mul']:
        shared_mlp_trainable = sum(p.numel() for p in model.shared_mlp.parameters() if p.requires_grad)
        shared_mlp_status = "✅ 凍結 (shared_mlp_flozen)" if shared_mlp_trainable == 0 else "✅ 学習オン (shared_mlp)"
        sanity_log.append(f"[3] 線形変換 (shared_mlp) : {shared_mlp_status} (学習可能パラメータ数: {shared_mlp_trainable})")

    elif args.temporal_emb_type in ['linear', 'linear_flozen']:
        time_mlp_trainable = sum(p.numel() for p in model.time_mlp.parameters() if p.requires_grad)
        linear_status = "✅ 凍結 (linear_flozen)" if time_mlp_trainable == 0 else "✅ 学習オン (linear)"
        sanity_log.append(f"[3] 線形変換 (time_mlp) : {linear_status} (学習可能パラメータ数: {time_mlp_trainable})")
        
    elif args.temporal_emb_type in [ 'positional','main']:
        # TPE (Temporal Positional Encoding) の場合は学習パラメータを持たない
        sanity_log.append(f"[3] 線形変換 (TPE) : ✅ 凍結 (Temporal Positional Encoding / 学習パラメータなし)")

    elif args.temporal_emb_type in ['MLP_token_renew_loss_ph_label','MLP_token_renew_vocab_topk_loss_ph','MLP_fusion','MLP_fusion_k','MLP_fusion_renew','MLP_token_renew_loss','MLP_token_renew_loss_p','MLP_token_renew_loss_h','MLP_token_renew_loss_ph','MLP_token_renew_topk_loss_p', 'MLP_token_renew_topk_loss_h', 'MLP_token_renew_topk_loss_ph']:
        fusion_mlp_trainable = sum(p.numel() for p in model.fusion_mlp.parameters() if p.requires_grad)
        fusion_mlp_status = "✅ 凍結 (shared_mlp_flozen)" if fusion_mlp_trainable == 0 else "✅ 学習オン (fusion_mlp)"
        sanity_log.append(f"[3] 線形変換 (fusion_mlp) : {fusion_mlp_status} (学習可能パラメータ数: {fusion_mlp_trainable})")

    # 4. 自然言語トークン・特殊トークンの確認
    if model.use_natural_separators:
        token_status_nat = "✅ 自然言語トークンを使用 (Time, Type, Magnitude, Depth)"
    else:
        token_status_nat = "✅ 特殊トークンを使用 (<|time_prefix|>, etc...)"
    sanity_log.append(f"[4] 使用するトークン    : {token_status_nat}")
    sanity_log.append(f"    - Temporal Emb Type : {model.temporal_emb_type}")
    
    # 5. トークン使用の有無
    if model.use_tokens:
        token_status = "✅ トークンを使用 "
    else:
        token_status = "✅ トークンを使用しない "
    sanity_log.append(f"[5] トークン有無        : {token_status}")
    
    # ⭕️ 追加: 総パラメータ数と学習可能パラメータ数の計算
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    sanity_log.append(f"[6] 総パラメータ数      : {total_params:,} (うち学習可能: {total_trainable:,})")

    # 🌟🌟🌟 ここに追加！ CodebookのShapeをログに記録 🌟🌟🌟
    if hasattr(model, 'codebook_p') and hasattr(model, 'codebook_h'):
        sanity_log.append(f"[7] Codebook (P) shape  : {model.codebook_p.shape}")
        sanity_log.append(f"[7] Codebook (H) shape  : {model.codebook_h.shape}")
    elif hasattr(model, 'codebook'):
        sanity_log.append(f"[7] Codebook shape      : {model.codebook.shape}")
    # 🌟🌟🌟 追加ここまで 🌟🌟🌟
    
    # =========================================================
    # 🌟 ここから追加: モデルの層数と各層のパラメータ数の詳細出力
    # =========================================================
    sanity_log.append("\n" + "-"*60)
    sanity_log.append(" 🔍 MODEL ARCHITECTURE DETAILS (Layer & Parameters)")
    sanity_log.append("-" * 60)
    
    # ① 層の数 (Layers) を Config から取得
    if hasattr(model.llm.config, "num_hidden_layers"):
        num_layers = model.llm.config.num_hidden_layers
        sanity_log.append(f"[*] Base Model Layers (Transformer Blocks): {num_layers} 層")
    else:
        sanity_log.append("[*] Base Model Layers: Configから自動取得できませんでした")

    # ② 最初の1層目（layers.0）のパラメータ構造だけをサンプルとして抽出（全部出すと長すぎるため）
    #sanity_log.append("\n[*] Layer 0 (最初の層) のパラメータ内訳:")
    #layer_0_params = 0
    #for name, param in model.llm.named_parameters():
     #   # "layers.0." が名前に含まれるものだけを抽出
      #  if "layers.0." in name:
       ##     num_p = param.numel()
         #   layer_0_params += num_p
     #       # LoRAが付与されている層かどうかのマーカー
      #      lora_mark = " (LoRA 🟢)" if "lora" in name else ""
       #     sanity_log.append(f"  - {name}: {num_p:,}{lora_mark}")
            
    #sanity_log.append(f"  >>> Layer 0 の合計パラメータ数: {layer_0_params:,}")

    # ③ LoRAの全学習対象パラメータの内訳
    #if args.peft_type == 'lora':
     #   sanity_log.append("\n[*] LoRA Adapter (学習対象) の内訳:")
     #   for name, param in model.llm.named_parameters():
     #       if "lora" in name and param.requires_grad:
     #           sanity_log.append(f"  - {name}: {param.numel():,}")
    # =========================================================

    # 既存のトークン埋め込み層のパラメータ数
    total_emb_params = model.llm_embedder.weight.numel()
# 更新対象外の古い語彙数分 (old_vocab_size * embedding_dim)
    frozen_emb_params = model.old_vocab_size * model.llm_embedder.embedding_dim
# 学習対象のトークン埋め込み数
    trainable_emb_params = total_emb_params - frozen_emb_params

# LoRAのパラメータ数（モデルがpeft_modelの場合）
    if hasattr(model.llm, 'print_trainable_parameters'):
    # PeftModelなら学習可能な総数からEmbedding分を引くか、個別に計算する
    # ここでは、model.parameters()からrequires_grad=Trueのものを合計する
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        actual_trainable = total_trainable - frozen_emb_params # 凍結分を差し引く
    else:
        actual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) - frozen_emb_params

    print(f"\n[Verified] 真の学習可能パラメータ数: {actual_trainable:,}")
    sanity_log.append(f"\n[Verified] 真の学習可能パラメータ数: {actual_trainable:,}")

    sanity_log.append("="*60 + "\n")
    
    # リストに貯めた文字列を結合
    full_sanity_text = "\n".join(sanity_log)
    
    # ターミナルに表示
    print(full_sanity_text)
    
    # val.txt に追記保存
    with open(val_log_path, 'a', encoding='utf-8') as f:
        f.write(full_sanity_text + "\n")
    # ---------------------------------------------------------

   # Inject dataset bounds (min/max) as model attributes for visualization & normalization
    model.max_lat = stats['lat']['max']
    model.max_lon = stats['lon']['max']
    model.max_dis = stats['dis']['max']
    model.max_time = stats['time']['max']
    model.min_lat = stats['lat']['min']
    model.min_lon = stats['lon']['min']
    model.min_dis = stats['dis']['min']
    model.min_time = stats['time']['min']
    model.min_delta = stats['delta']['min']
    model.max_delta = stats['delta']['max']

    model.mean_lat = stats['lat']['mean']
    model.std_lat = stats['lat']['std']
    model.mean_lon = stats['lon']['mean']
    model.std_lon = stats['lon']['std']
    model.mean_dis = stats['dis']['mean']
    model.std_dis = stats['dis']['std']
    model.mean_time = stats['time']['mean']
    model.std_time = stats['time']['std']
    
    
    # ---------------------------------------------------------
    # ★ 修正: 正規化の基準値（Mean/Std と Min/Max）を画面とログファイルに記録
    # ---------------------------------------------------------
    stats_log = f"""{'='*60}
 📊 DATASET STATISTICS (Standardization & Min-Max){'='*60}
  - Time  : Mean={model.mean_time:.4f}, Std={model.std_time:.4f} (Min/Max: {model.min_time:.4f} ~ {model.max_time:.4f})
  - Latitude   : Mean={model.mean_lat:.4f}, Std={model.std_lat:.4f} (Min/Max: {model.min_lat:.4f} ~ {model.max_lat:.4f})
  - Longitude : Mean={model.mean_lon:.4f}, Std={model.std_lon:.4f} (Min/Max: {model.min_lon:.4f} ~ {model.max_lon:.4f})
  - Distance : Mean={model.mean_dis:.4f}, Std={model.std_dis:.4f} (Min/Max: {model.min_dis:.4f} ~ {model.max_dis:.4f})
  - Delta : Mean={stats['delta']['mean']:.4f}, Std={stats['delta']['std']:.4f} (Min/Max: {model.min_delta:.4f} ~ {model.max_delta:.4f})
    {'='*60}\n"""
    
    # ターミナルに表示
    print(stats_log)
    
    # val.txt に追記保存
    with open(val_log_path, 'a', encoding='utf-8') as f:
        f.write(stats_log)
    # ---------------------------------------------------------
    
    # Training/Evaluation Execution
    print("Starting Runner...")
    runner = TPPLLMRunner(
        model=model,
        beta_type=args.beta_type,
        beta_time=args.beta_time,
        device=args.device,
        weight_path=args.weight_path,
        load_flag=args.load_flag,
    )
    # ==========================================================
    # 🚨 追加・変更：エラーをキャッチしてログに書き込む！
    # ==========================================================
    try:
        runner.run(
            quantitative_analysis_func=perform_quantitative_analysis,
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            dataloader_test=dataloader_test,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            num_train_epochs=args.num_epochs,
            warmup_ratio=args.warmup_ratio,
            result_save_path=args.result_save_path,
            weight_path=args.weight_path,
            model_weight_path=args.model_weight_path,
            save_flag=args.save_flag,
            train_flag=args.train_flag,
            resume_flag=args.resume_flag, # ⬅️ これを追加
            gradient_accumulation_steps=4,
            temporal_emb_type=args.temporal_emb_type
        )

    except Exception as e:
        # エラーの完全な詳細（Traceback）を取得
        error_msg = f"\n{'='*60}\n 🚨 FATAL ERROR CRASH 🚨\n{'='*60}\n{traceback.format_exc()}\n{'='*60}\n"
        print(error_msg)
        # val.txt に強制書き込み
        with open(val_log_path, 'a', encoding='utf-8') as f:
            f.write(error_msg)
    finally:
        # ⭕️ 追加：正常終了でもエラーでも「絶対に」実行されるブロック
        end_time = time.time()
        execution_time = end_time - start_time
        
        time_msg = f"\n{'='*60}\n⏱️ Total Execution Time: {execution_time:.2f} seconds\n{'='*60}\n"
        print(time_msg)
        
        with open(val_log_path, 'a', encoding='utf-8') as f:
            f.write(time_msg)
    # ==========================================================