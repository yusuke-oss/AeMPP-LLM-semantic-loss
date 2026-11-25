"""
TPP-LLM Runner
Handles the training loop, evaluation, and gradient accumulation.
"""
import os
import json
from typing import Dict, Tuple, List, Union
from sklearn.metrics import cohen_kappa_score

import numpy as np
import torch
import re          # 🌟 追加
import ast         # 🌟 追加
import matplotlib.pyplot as plt # 🌟 追加
import transformers
import time  # ⭕️ 追加
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support
import random  # ⬅️ これを追加
from src.tpp_llm.earthquake.model import TPPLLMModel

class TPPLLMRunner(object):
    """
    TPP-LLM Runner
    """

    def __init__(
        self, model: TPPLLMModel, beta_type: float = 1, beta_time: float = 1,beta_ll:float=1,device: Union[str, torch.device] = 'cpu'
        ,weight_path:str='',load_flag:bool=False):
        """
        Initialize the TPP-LLM runner.

        :param model: The TPPLLMModel instance.
        :param beta_type: Loss coefficient for event type prediction.
        :param beta_time: Loss coefficient for event time prediction.
        :param beta_semantic: Loss coefficient for semantic alignment.
        :param device: Execution device.
        """


        self.model = model
        self.beta_type = beta_type
        self.beta_time = beta_time
        self.beta_ll=beta_ll
        print(beta_ll)
        print(beta_type)
        print(beta_time)
        self.device = torch.device(device)

        self.num_train_epochs = 0
        self.num_training_steps = 0
        self.num_warmup_steps = 0
        self.global_step = 0
        self.scheduler = None
        self.optimizer = None

        if(load_flag):
            self.load(weight_path)

        # Lambda functions to detach tensors and move them to CPU as numpy arrays
        self.move_to_numpy_prev = lambda list_tensors: [tensor[1:].numpy(force=True) for tensor in list_tensors]
        self.move_to_numpy_next = lambda list_tensors: [tensor[:-1].numpy(force=True) for tensor in list_tensors]

    def run_batch(self, batch: Dict[str, list], phase: str) \
        -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray],List[np.ndarray]]:
        """
        Run a single batch of event sequences.

        Returns:
            Tuple containing:
            - event_nums: Number of events per sequence
            - log_likelihoods: Log-likelihoods of the sequences
            - event_types_shifted: True event types (shifted for prediction)
            - next_event_types_shifted: Predicted next event types
            - event_times_shifted: True event times (shifted)
            - next_event_times_shifted: Predicted next event times
            - event_time_deltas_shifted: True time deltas (shifted)
            - semantic_loss: Semantic alignment loss (float)
            - batch_loss: Total combined loss (Tensor, for backward pass)
            - event_mags_shifted: True event magnitudes (shifted)
        """

        # Move inputs to the target device
        batch = {
            'time_since_start': [_seq.to(self.device) for _seq in batch['time_since_start']],
            'time_since_last_event': [_seq.to(self.device) for _seq in batch['time_since_last_event']],
            'type_event': [_seq.to(self.device) for _seq in batch['type_event']],
            'type_text':batch['type_text'],
            'magnitude':[_seq.to(self.device) for _seq in batch['magnitude']],
            'depth':[_seq.to(self.device) for _seq in batch['depth']],
            'latitude':[_seq.to(self.device) for _seq in batch['latitude']],
            'longitude':[_seq.to(self.device) for _seq in batch['longitude']],
            'time_float':batch['time_float'],
            'magnitude_float':batch['magnitude_float'],
            'depth_float':batch['depth_float'],
            'latitude_float':batch['latitude_float'],
            'longitude_float':batch['longitude_float'],
        }

        if phase == 'train':
            self.model.train()

            batch_event_nums, batch_nll_losses, batch_type_losses, batch_time_losses, batch_next_event_types, batch_next_event_times, codebook_losses, batch_words_p, batch_words_h, batch_distination_p, batch_distination_h, batch_dist_eq_p, batch_dist_eq_h, raw_cb_p, raw_cb_h ,temp_embs_for_loss,batch_hidden_states= self.model.compute_loss(batch)
            # 1. 基本のタスク損失（系列ごとの配列）
            base_losses = self.beta_ll*batch_nll_losses + self.beta_type * batch_type_losses + self.beta_time * batch_time_losses
            batch_type_loss = torch.sum(batch_type_losses)
            batch_time_loss = torch.sum(batch_time_losses)
            batch_nll_loss = torch.sum(batch_nll_losses)
            # 2. 🌟 Codebook loss が存在する場合は重みをかけて加算する
            if codebook_losses is not None:
                total_losses_per_seq = base_losses +  codebook_losses
                # 最後に全体を合計する
                batch_loss = torch.sum(total_losses_per_seq)
                
                # ログ・コンソール出力用のスカラー値（ペナルティ適用後の合計値）
                loss_codebook_val = torch.sum( codebook_losses).item()
            else:
                batch_loss = torch.sum(base_losses)
                loss_codebook_val = 0.0
        
            batch_event_nums = batch_event_nums.numpy(force=True)
            batch_log_likelihoods = - batch_nll_losses.numpy(force=True)
            

            metrics = {
                'batch_loss': batch_loss.float().cpu().item(),
                'loss_tpp': torch.sum(batch_nll_losses).float().cpu().item(),
                'loss_type': torch.sum(batch_type_losses).float().cpu().item(),
                'loss_time': torch.sum(batch_time_losses).float().cpu().item(),
                # semantic_loss は削除
                'loss_codebook': loss_codebook_val,  # 🌟 コンソール出力・ログに追加
                'raw_P_loss': torch.sum(raw_cb_p).item() if raw_cb_p is not None else 0.0, # 🌟 コンソール用
                'raw_H_loss': torch.sum(raw_cb_h).item() if raw_cb_h is not None else 0.0, # 🌟 コンソール用
                'batch_event_nums': batch_event_nums.sum().item(),
                'batch_log_likelihood': batch_log_likelihoods.sum().item() / batch_event_nums.sum().item(),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch': self.global_step / self.num_training_steps * self.num_train_epochs,
            }
            print(metrics)
            # 🌟 ここを変更: Trainでも予測結果をNumpy配列に変換して返す
            batch_event_types_shifted = self.move_to_numpy_prev(batch['type_event'])
            batch_event_times_shifted = self.move_to_numpy_prev(batch['time_since_start'])
            batch_next_event_types_shifted = self.move_to_numpy_next(batch_next_event_types)
            batch_next_event_times_shifted = self.move_to_numpy_next(batch_next_event_times)
            batch_event_time_deltas_shifted = self.move_to_numpy_prev(batch['time_since_last_event'])

            # 戻り値を11個に増やす（15個受け取って必要なものを返す）
            return batch_event_nums, batch_log_likelihoods, batch_event_types_shifted, \
                   batch_next_event_types_shifted, batch_event_times_shifted, \
                   batch_next_event_times_shifted, batch_event_time_deltas_shifted, \
                   batch_loss, batch_words_p, batch_words_h, batch_distination_p, batch_distination_h,batch_dist_eq_p, batch_dist_eq_h, raw_cb_p, raw_cb_h,temp_embs_for_loss,batch_hidden_states,batch_type_loss ,batch_time_loss,batch_nll_loss

        elif phase == 'eval':
            self.model.eval()

            batch_event_times = batch['time_since_start']
            batch_event_types = batch['type_event']
            batch_event_time_deltas = batch['time_since_last_event']

            # ========== 変更後 ==========
            with torch.no_grad():
                batch_event_nums, batch_nll_losses, batch_type_losses, batch_time_losses, batch_next_event_types, batch_next_event_times, codebook_losses, batch_words_p, batch_words_h, batch_distination_p, batch_distination_h, batch_dist_eq_p, batch_dist_eq_h, raw_cb_p, raw_cb_h ,temp_embs_for_loss,batch_hidden_states= self.model.compute_loss(batch)
                
                base_losses =self.beta_ll* batch_nll_losses + self.beta_type * batch_type_losses + self.beta_time * batch_time_losses

                # シーケンス全体の合計を出してスカラーに変換
                batch_loss = torch.sum(base_losses).float().cpu().item()
                batch_type_loss = torch.sum(batch_type_losses).float().cpu().item()
                batch_time_loss = torch.sum(batch_time_losses).float().cpu().item()
                batch_nll_loss = torch.sum(batch_nll_losses).float().cpu().item()
                
                batch_log_likelihoods = -batch_nll_losses

            # Shift and convert tensors
            batch_event_nums = batch_event_nums.numpy(force=True)
            batch_log_likelihoods = batch_log_likelihoods.numpy(force=True)
            batch_event_types_shifted = self.move_to_numpy_prev(batch_event_types)
            batch_event_times_shifted = self.move_to_numpy_prev(batch_event_times)
            batch_next_event_types_shifted = self.move_to_numpy_next(batch_next_event_types)
            batch_next_event_times_shifted = self.move_to_numpy_next(batch_next_event_times)
            batch_event_time_deltas_shifted = self.move_to_numpy_prev(batch_event_time_deltas)
            
            # 戻り値を変更 
            return batch_event_nums, batch_log_likelihoods, batch_event_types_shifted, \
                batch_next_event_types_shifted, batch_event_times_shifted, \
                batch_next_event_times_shifted, batch_event_time_deltas_shifted, \
                batch_loss, batch_words_p, batch_words_h, batch_distination_p, batch_distination_h,batch_dist_eq_p, batch_dist_eq_h, raw_cb_p, raw_cb_h,temp_embs_for_loss,batch_hidden_states,batch_type_loss,batch_time_loss,batch_nll_loss

        else:
            raise KeyError(f'Unknown phase: {phase}.')

    def run_epoch(self, dataloader: DataLoader, phase: str, result_save_path:str, gradient_accumulation_steps: int = 1) -> dict:
        """
        Run an epoch (Updated for Gradient Accumulation)
        :param gradient_accumulation_steps: 勾配蓄積数 (デフォルト1)
        """

        total_log_likelihood = 0
        total_num_events = 0
        total_val_loss = 0.0  
        total_train_loss = 0.0  # 🌟 追加 (Train時のLoss集計用)
        total_val_type_loss = 0.0 
        total_val_time_loss = 0.0  
        total_val_nll_loss = 0.0  
        total_train_type_loss = 0.0  # 🌟 追加 (Train時のLoss集計用)
        total_train_time_loss = 0.0  # 🌟 追加 (Train時のLoss集計用)
        total_train_nll_loss = 0.0  # 🌟 追加 (Train時のLoss集計用)
        total_raw_p_loss = 0.0  # 🌟 追加 (生のP層ロス集計用)
        total_raw_h_loss = 0.0  # 🌟 追加 (生のH層ロス集計用)
        num_batches = 0
        
        raw_event_types = []
        raw_event_type_preds = []
        raw_event_times = []
        raw_event_time_preds = []
        raw_event_time_deltas = []
        
        all_event_types = []
        all_event_type_preds = []
        all_event_times = []
        all_event_time_preds = []
        all_event_time_deltas = []
        
        raw_event_data_list = []
        all_seq_ll = []  # 🌟 追加：系列ごとの対数尤度を保存
        all_inf_times = []
        metrics = {}

        # ★ Trainフェーズ開始時に勾配を初期化
        if phase == 'train':
            self.optimizer.zero_grad()

        # ★ enumerate でステップ数を取得
        total_inference_time = 0.0  # ⭕️ 追加: 推論時間の合計用変数

        # ★ enumerate でステップ数を取得
        for step, batch in enumerate(tqdm(dataloader)):

            # ⭕️ 追加: evalフェーズのみ時間計測を開始
            
            start_inf_time = time.time()

            # 🌟 16個の値を受け取る
            event_nums, log_likelihoods, event_types, event_type_preds, event_times, \
            event_time_preds, event_time_deltas, batch_loss_tensor, batch_words_p, batch_words_h, batch_distination_p, batch_distination_h, batch_dist_eq_p, batch_dist_eq_h, raw_cb_p, raw_cb_h ,temp_embs_for_loss,batch_hidden_states,batch_type_loss,batch_time_loss,batch_nll_loss= \
            self.run_batch(batch=batch, phase=phase)

            batch_inf_time = time.time() - start_inf_time
            total_inference_time += batch_inf_time

            # 🌟 生ロスの集計 (テンソルが返ってきている場合)
            if raw_cb_p is not None:
                total_raw_p_loss += float(torch.sum(raw_cb_p).cpu().item())
            if raw_cb_h is not None:
                total_raw_h_loss += float(torch.sum(raw_cb_h).cpu().item())

            # ⭕️ 追加: 1バッチの推論が終わったら時間を足す
            if phase == 'eval':
                total_val_loss += batch_loss_tensor # 🌟 追加 (Val_lossの集計)
                total_val_type_loss += batch_type_loss # 🌟 追加 (Val_lossの集計)
                total_val_time_loss += batch_time_loss # 🌟 追加 (Val_lossの集計)
                total_val_nll_loss += batch_nll_loss # 🌟 追加 (Val_lossの集計)
                # 🌟 追加: 推論時間のばらつき（標準偏差）を記録
            batch_events = np.sum(event_nums)
            if batch_events > 0:
                time_per_event_ms = (batch_inf_time / batch_events) * 1000
                # バッチ内の系列すべてに、この平均処理時間を記録
                all_inf_times.append(time_per_event_ms)

            # 変更後
            total_log_likelihood += np.sum(log_likelihoods)
            total_num_events += np.sum(event_nums)
            
            # 🌟 追加：系列ごとの対数尤度を保存
            if len(event_nums) > 0:
                seq_lls = log_likelihoods / event_nums
                all_seq_ll.extend(seq_lls.tolist())
            
            if phase == 'train':
                num_batches += 1  # total_semantic_loss += semantic_loss は削除
                total_train_loss += float(batch_loss_tensor) # 🌟 追加 (floatで値だけ取り出して足すことでメモリリーク防止)
                total_train_type_loss += float(batch_type_loss)
                total_train_time_loss += float(batch_time_loss)
                total_train_nll_loss += float(batch_nll_loss)
                # 1. Average the loss over the accumulation steps
                loss = batch_loss_tensor / gradient_accumulation_steps
                
                # 2. Backward pass (accumulate gradients)
                loss.backward()
                
                # 3. Update parameters if we reached the accumulation step limit or the end of the dataloader
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    
                    # Zero out the gradients of the original LLM embeddings to keep them frozen
                    emb = self.model.llm.get_input_embeddings()
                    if emb.weight.grad is not None:
                        emb.weight.grad[:self.model.old_vocab_size].zero_()

                    if emb.weight.grad is not None:
                        nonzero = (emb.weight.grad.abs().sum(dim=1) > 0).sum().item()
                        print(f"[grad-check] nonzero embedding rows: {nonzero} (should equal num_added_tokens)")
            

                    
                    # Update the parameters
                    self.optimizer.step()
                    
                    # Update the learning rate
                    self.scheduler.step()
                    
                    # Optimize the model parameters
                    self.optimizer.zero_grad()
                    
                    # Update the Global Step
                    self.global_step += 1

                    
            
                    

                # ========================================================

            
            

            # 🌟 追加: オリジナルのバッチから補助情報をNumpyとして取り出しておく
            batch_mags = [m.cpu().numpy() for m in batch['magnitude']]
            batch_deps = [d.cpu().numpy() for d in batch['depth']]
            batch_lats = [l.cpu().numpy() for l in batch['latitude']]
            batch_lons = [l.cpu().numpy() for l in batch['longitude']]
            batch_event_time_t= [l.cpu().numpy() for l in batch['time_since_start']]
            batch_event_type_t= [l.cpu().numpy() for l in batch['type_event']]
            # runner.py 内
            batch_dist_eq_p = locals().get('batch_dist_eq_p', [])
            batch_dist_eq_h = locals().get('batch_dist_eq_h', [])
            
            
            # ============================================================
            # raw_data 保存
            # ============================================================
            if phase == 'eval':
            
                for i in range(len(event_types)):

                    # --------------------------------------------
                    # Input vector
                    # --------------------------------------------
                    if (
                        temp_embs_for_loss
                        and i < len(temp_embs_for_loss)
                    ):
                        input_vector = (
                            temp_embs_for_loss[i]
                            .numpy()
                            .tolist()
                        )
                    else:
                        input_vector = []

                    # --------------------------------------------
                    # Hidden vector
                    # --------------------------------------------
                    if (
                        batch_hidden_states
                        and i < len(batch_hidden_states)
                    ):
                        hidden_vector = (
                            batch_hidden_states[i]
                            .numpy()
                            .tolist()
                        )
                    else:
                        hidden_vector = []

                    # --------------------------------------------
                    # raw_data
                    # --------------------------------------------
                    raw_event_data_list.append({

                        'true_types': (
                            event_types[i].tolist()
                            if isinstance(
                                event_types[i],
                                np.ndarray
                            )
                            else event_types[i]
                        ),

                        'pred_types': (
                            event_type_preds[i].tolist()
                            if isinstance(
                                event_type_preds[i],
                                np.ndarray
                            )
                            else event_type_preds[i]
                        ),

                        'true_times': (
                            event_times[i].tolist()
                            if isinstance(
                                event_times[i],
                                np.ndarray
                            )
                            else event_times[i]
                        ),

                        'pred_times': (
                            event_time_preds[i].tolist()
                            if isinstance(
                                event_time_preds[i],
                                np.ndarray
                            )
                            else event_time_preds[i]
                        ),

                        'mags': (
                            batch_mags[i][0:].tolist()
                            if i < len(batch_mags)
                            else []
                        ),

                        'deps': (
                            batch_deps[i][0:].tolist()
                            if i < len(batch_deps)
                            else []
                        ),

                        'lats': (
                            batch_lats[i][0:].tolist()
                            if i < len(batch_lats)
                            else []
                        ),

                        'lons': (
                            batch_lons[i][0:].tolist()
                            if i < len(batch_lons)
                            else []
                        ),

                        'time': (
                            batch_event_time_t[i][0:].tolist()
                            if i < len(batch_event_time_t)
                            else []
                        ),

                        'type': (
                            batch_event_type_t[i][0:].tolist()
                            if i < len(batch_event_type_t)
                            else []
                        ),

                        'tsne_type': (
                            batch_event_type_t[i].tolist()
                            if i < len(batch_event_type_t)
                            else []
                        ),

                        'tsne_time': (
                            batch_event_time_t[i].tolist()
                            if i < len(batch_event_time_t)
                            else []
                        ),

                        'words_p': (
                            batch_words_p[i][0:]
                            if (
                                batch_words_p
                                and i < len(batch_words_p)
                            )
                            else []
                        ),

                        'words_h': (
                            batch_words_h[i][0:]
                            if (
                                batch_words_h
                                and i < len(batch_words_h)
                            )
                            else []
                        ),

                        'dist_p': (
                            batch_distination_p[i][0:]
                            if (
                                batch_distination_p
                                and i < len(batch_distination_p)
                            )
                            else []
                        ),

                        'dist_h': (
                            batch_distination_h[i][0:]
                            if (
                                batch_distination_h
                                and i < len(batch_distination_h)
                            )
                            else []
                        ),

                        'dist_eq_p': (
                            batch_dist_eq_p[i][0:]
                            if (
                                batch_dist_eq_p
                                and i < len(batch_dist_eq_p)
                            )
                            else []
                        ),

                        'dist_eq_h': (
                            batch_dist_eq_h[i][0:]
                            if (
                                batch_dist_eq_h
                                and i < len(batch_dist_eq_h)
                            )
                            else []
                        ),

                        # ====================================================
                        # ★ t-SNE 用 Input / Hidden vector
                        # ====================================================
                        'input_vector': input_vector,

                        'hidden_vector': hidden_vector,
                    })
                
            all_event_types.append(np.concatenate(event_types))
            all_event_type_preds.append(np.concatenate(event_type_preds))
            all_event_times.append(np.concatenate(event_times))
            all_event_time_preds.append(np.concatenate(event_time_preds))
            all_event_time_deltas.append(np.concatenate(event_time_deltas))

            raw_event_types.extend(event_types)
            raw_event_type_preds.extend(event_type_preds)
            raw_event_times.extend(event_times)
            raw_event_time_preds.extend(event_time_preds)
            raw_event_time_deltas.extend(event_time_deltas)

        # Calculate final metrics for the epoch
        avg_log_likelihood = total_log_likelihood / total_num_events
        metrics['log_likelihood'] = float(avg_log_likelihood)
        metrics['num_events'] = int(total_num_events)
        
        if phase == 'train' and num_batches > 0:
            # 🌟 追加 (全イベントで割って、1イベントあたりの平均Train Lossを算出)
            metrics['train_loss'] = float(total_train_loss / total_num_events) if total_num_events > 0 else 0.0
            metrics['train_type_loss'] = float(total_train_type_loss / total_num_events) if total_num_events > 0 else 0.0
            metrics['train_time_loss'] = float(total_train_time_loss / total_num_events) if total_num_events > 0 else 0.0
            metrics['train_nll_loss'] = float(total_train_nll_loss / total_num_events) if total_num_events > 0 else 0.0
        elif phase == 'eval':
            metrics['val_loss'] = float(total_val_loss / total_num_events) if total_num_events > 0 else 0.0
            metrics['val_type_loss'] = float(total_val_type_loss / total_num_events) if total_num_events > 0 else 0.0
            metrics['val_time_loss'] = float(total_val_time_loss / total_num_events) if total_num_events > 0 else 0.0
            metrics['val_nll_loss'] = float(total_val_nll_loss / total_num_events) if total_num_events > 0 else 0.0
        # 🌟 生のLossの平均も epoch 全体の metrics に保存する (Train/Eval/Test 共通)
        metrics['raw_P_loss'] = float(total_raw_p_loss / total_num_events) if total_num_events > 0 else 0.0
        metrics['raw_H_loss'] = float(total_raw_h_loss / total_num_events) if total_num_events > 0 else 0.0
        
        all_event_types = np.concatenate(all_event_types)
        all_event_type_preds = np.concatenate(all_event_type_preds)
        all_event_times = np.concatenate(all_event_times)
        all_event_time_preds = np.concatenate(all_event_time_preds)
        all_event_time_deltas = np.concatenate(all_event_time_deltas)

        accuracy = np.mean(all_event_types == all_event_type_preds)
        rmse = np.sqrt(np.mean((all_event_times - all_event_time_preds) ** 2))

            # ★ ここに追加
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_event_types, 
            all_event_type_preds, 
            labels=[0,1,2],
            average='macro', 
            zero_division=0 # クラスによっては予測がない場合にエラーを防ぐため
        )
        from collections import Counter

        print("True class counts:", Counter(all_event_types))
        print("Pred class counts:", Counter(all_event_type_preds))
        print("Confusion matrix:")
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(
            all_event_types,
            all_event_type_preds,
            labels=[0, 1, 2]
        ))
        # 全イベントの Quadratic Weighted Kappa
        qwk = cohen_kappa_score(
            all_event_types,
            all_event_type_preds,
            labels=[0, 1, 2],
            weights='quadratic'
        )
            
        metrics['accuracy'] = float(accuracy)
        metrics['rmse'] = float(rmse)
        metrics['precision'] = float(precision) # 追加
        metrics['recall'] = float(recall)       # 追加
        metrics['f1'] = float(f1)               # 追加
        metrics['qwk'] = float(qwk)
            # ⭕️ 追加: 1イベントあたりの平均推論時間をミリ秒（ms）で計算
        avg_inf_time_ms = (total_inference_time / total_num_events) * 1000 if total_num_events > 0 else 0
        metrics['avg_inf_time_per_event(ms)'] = float(avg_inf_time_ms)
            
            
        seq_accuracies = []
        seq_rmses = []
        seq_f1s = []
        seq_qwks = []

        for true_typ, pred_typ, true_tm, pred_tm in zip(
            raw_event_types, raw_event_type_preds, raw_event_times, raw_event_time_preds
        ):
            seq_acc = np.mean(true_typ == pred_typ)
            seq_accuracies.append(float(seq_acc))
            seq_rmse = np.sqrt(np.mean((true_tm - pred_tm) ** 2))
            seq_rmses.append(float(seq_rmse))
            # 系列ごとの F1スコア を計算
            _, _, seq_f1, _ = precision_recall_fscore_support(true_typ, pred_typ,labels=[0,1,2], average='macro', zero_division=0)
            seq_f1s.append(float(seq_f1))
            # 系列ごとの QWK
            seq_qwk = cohen_kappa_score(
                true_typ,
                pred_typ,
                labels=[0, 1, 2],
                weights='quadratic'
            )
            seq_qwks.append(float(seq_qwk))
            
        # 辞書に標準偏差を登録
        metrics['log_likelihood_std'] = float(np.std(all_seq_ll)) if len(all_seq_ll) > 0 else 0.0
        metrics['rmse_std'] = float(np.std(seq_rmses)) if len(seq_rmses) > 0 else 0.0
        metrics['f1_std'] = float(np.std(seq_f1s)) if len(seq_f1s) > 0 else 0.0
        metrics['qwk_std'] = (
            float(np.nanstd(seq_qwks))
            if np.any(np.isfinite(seq_qwks)) else 0.0
        )
        metrics['inf_time_std(ms)'] = float(np.std(all_inf_times)) if len(all_inf_times) > 0 else 0.0
        # ====================================================

        seq_scores = {
            "accuracies": seq_accuracies,
           "rmses": seq_rmses,
            "f1s": seq_f1s, # これも追加しておくと後で分析しやすいです
            "lls": all_seq_ll,  # 🌟 この1行を追加
            "raw_data": raw_event_data_list
        }
        # 🌟 呼び出し元（runメソッド）の形を崩さないよう、evalの時だけ tuple を返す
        if phase == 'eval':
            return metrics, (all_event_types, all_event_type_preds,
                             all_event_times, all_event_time_preds,
                             all_event_time_deltas, seq_scores)

        # train の時は metrics のみを返す
        return metrics

    def run(
        self, quantitative_analysis_func: callable,
        dataloader_train: DataLoader = None, dataloader_val: DataLoader = None,
        dataloader_test: DataLoader = None, learning_rate: float = 5e-4, lr_scheduler_type: str = 'constant',
        num_train_epochs: int = 1, warmup_ratio: float = 0, result_save_path: str = '',
        weight_path: str = '', model_weight_path: str = '', figure_path: str = '', save_flag: bool = False, train_flag: bool = False,
        gradient_accumulation_steps: int = 1,temporal_emb_type: str=""
    ) -> None:
        """
        Run the training (Updated)
        """
        # Calculate steps considering accumulation
        self.num_train_epochs = num_train_epochs
        print(gradient_accumulation_steps)
        steps_per_epoch = len(dataloader_train) // gradient_accumulation_steps
        self.num_training_steps = self.num_train_epochs * steps_per_epoch
        self.num_warmup_steps = int(warmup_ratio * self.num_training_steps)
        self.global_step = 0
        self.temporal_emb_type=temporal_emb_type
        
        self.optimizer = Adam(
            list(self.model.parameters()) ,
            lr=learning_rate
        )

        if lr_scheduler_type == 'constant':
            self.scheduler = transformers.get_constant_schedule(optimizer=self.optimizer)
        elif lr_scheduler_type == 'constant_with_warmup':
            self.scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps)
        elif lr_scheduler_type == 'linear':
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)
        elif lr_scheduler_type == 'cosine':
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps, num_cycles=0.5)
        else:
            raise KeyError(f'Unknown learning rate scheduler type: {lr_scheduler_type}')

        metrics_val_best = None

        start_epoch = 0

        # Initial validation
        if dataloader_val:
            # 🌟🌟🌟 修正：run_epochを実行する【前】にリセットとヘッダー書き込みを行う 🌟🌟🌟
            if hasattr(self.model, 'reset_topk_debug_flags'):
                self.model.reset_topk_debug_flags()
                self.model.result_save_path = result_save_path  # モデルに保存先パスを教える
                try:
                    log_file_path = f"{result_save_path}/topk_debug_log.txt"
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(f"\n{'='*50}\n 🚀 Initial の 検証 Top-k 単語観測 (Validation)\n{'='*50}\n")
                except Exception:
                    pass
            metrics_val, eval_data = self.run_epoch(dataloader_val, phase='eval', result_save_path=result_save_path)
            print(f'validation metrics: {metrics_val}')
            metrics_val_best = metrics_val
            with open(result_save_path+'/val.txt', 'a') as f:
                f.write(f'validation metrics: {metrics_val}\n')
            
            
            (all_types_true, all_types_pred, all_times_true, all_time_preds, all_time_deltas_true, seq_scores_val) = eval_data
            if self.temporal_emb_type in ['MLP_token_renew_vocab_topk_loss_ph','MLP_token_renew_vocab_loss_ph_scaled','MLP_token_renew_loss', 'MLP_token_renew_loss_p', 'MLP_token_renew_loss_h','MLP_token_renew_loss_ph','MLP_token_renew_topk_loss_p', 'MLP_token_renew_topk_loss_h', 'MLP_token_renew_topk_loss_ph','MLP_token_renew_vocab_loss_ph']:
            
                quantitative_analysis_func(
                    all_types_true, all_types_pred, all_times_true, all_time_preds, all_time_deltas_true,
                    result_save_path, "initial_val","initial",self.model, seq_scores_val
                )
            
        # ---------------------------------------------------------
        # Initial Test
        # ---------------------------------------------------------
        if dataloader_test:
            # 🌟🌟🌟 修正：run_epochを実行する【前】にリセットとヘッダー書き込みを行う 🌟🌟🌟
            if hasattr(self.model, 'reset_topk_debug_flags'):
                self.model.reset_topk_debug_flags()
                self.model.result_save_path = result_save_path  # モデルに保存先パスを教える
                try:
                    log_file_path = f"{result_save_path}/topk_debug_log.txt"
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(f"\n{'='*50}\n 🚀 Initial の テスト Top-k 単語観測 (Validation)\n{'='*50}\n")
                except Exception:
                    pass
            metrics_test, test_eval_data = self.run_epoch(dataloader_test, phase='eval', result_save_path=result_save_path)
            print(f'test metrics: {metrics_test}')
            with open(result_save_path+'/test.txt', 'a') as f:
                f.write(f'test metrics: {metrics_test}\n')

            (all_types_true_test, all_types_pred_test, all_times_true_test, all_time_preds_test, all_time_deltas_true_test, seq_scores_test) = test_eval_data
            if self.temporal_emb_type in ['MLP_token_renew_vocab_topk_loss_ph','MLP_token_renew_vocab_loss_ph_scaled','MLP_token_renew_loss', 'MLP_token_renew_loss_p', 'MLP_token_renew_loss_h','MLP_token_renew_loss_ph','MLP_token_renew_topk_loss_p', 'MLP_token_renew_topk_loss_h', 'MLP_token_renew_topk_loss_ph','MLP_token_renew_vocab_loss_ph']:
                        
                quantitative_analysis_func(
                    all_types_true_test, all_types_pred_test, all_times_true_test, all_time_preds_test, all_time_deltas_true_test,
                    result_save_path, "initial_test","initial", self.model,seq_scores_test
                )
        
        patience = 10
        epochs_no_improve = 0

        # ループを start_epoch から始まるように変更
        for epoch in range(start_epoch, int(num_train_epochs)):

            
            if save_flag or train_flag:
                print(f'epoch: {epoch}')
            
            if dataloader_train and (save_flag or train_flag):
                metrics_train = self.run_epoch(dataloader_train, phase='train', result_save_path=result_save_path, 
                                               gradient_accumulation_steps=gradient_accumulation_steps)
                if(epoch==0):
                    import subprocess
                    with open(result_save_path+'/val.txt', 'a') as f:
                        subprocess.run(["nvidia-smi"], stdout=f, text=True)
                
                print(f'train metrics of epoch {epoch}: {metrics_train}')
                with open(result_save_path+'/train.txt', 'a') as f:
                    f.write(f'train metrics of epoch {epoch}: {metrics_train}\n')

            if dataloader_val and (save_flag or train_flag):

                # 🌟🌟🌟 ここにお引越し：Valの直前にリセットし、ヘッダーを書く 🌟🌟🌟
                if hasattr(self.model, 'reset_topk_debug_flags'):
                    self.model.reset_topk_debug_flags()
                    self.model.result_save_path = result_save_path  # 👈【追加】モデルに保存先パスを教える
                    try:
                        # 👈【変更】保存先を result_save_path の中にする
                        log_file_path = f"{result_save_path}/topk_debug_log.txt"
                        with open(log_file_path, "a", encoding="utf-8") as f:
                            f.write(f"\n{'='*50}\n 🚀 Epoch {epoch} の 検証 Top-k 単語観測 (Validation)\n{'='*50}\n")
                    except Exception:
                        pass
                # 🌟🌟🌟 追加ここまで 🌟🌟🌟
                
                metrics_val, eval_data = self.run_epoch(dataloader_val, phase='eval', result_save_path=result_save_path)
                print(f'validation metrics of epoch {epoch}: {metrics_val}')

                with open(result_save_path+'/val.txt', 'a') as f:
                    f.write(f'validation metrics of epoch {epoch}: {metrics_val}\n')

                    # 判定を「val_lossが過去最小を下回ったか」に変更！
                    if metrics_val_best is None or metrics_val['val_loss'] < metrics_val_best.get('val_loss', float('inf')):
                        metrics_val_best = metrics_val
                        metrics_val_best['val_loss'] = metrics_val['val_loss']
                        epochs_no_improve = 0
                        print(f'new best validation metrics (by Validation Loss)')
                        f.write(f'new best validation metrics (by Validation Loss)\n')

                        if save_flag:
                            # ⬇️ 修正: 引数に epoch と metrics_val_best を追加
                            self.save(model_weight_path, weight_path, epoch=epoch, metrics_val_best=metrics_val_best)

                        (all_types_true, all_types_pred, all_times_true, all_time_preds, all_time_deltas_true, seq_scores_val) = eval_data
                        if self.temporal_emb_type in ['MLP_token_renew_vocab_topk_loss_ph','MLP_token_renew_vocab_loss_ph_scaled','MLP_token_renew_loss', 'MLP_token_renew_loss_p', 'MLP_token_renew_loss_h','MLP_token_renew_loss_ph','MLP_token_renew_topk_loss_p', 'MLP_token_renew_topk_loss_h', 'MLP_token_renew_topk_loss_ph','MLP_token_renew_vocab_loss_ph']:
                                    
                            quantitative_analysis_func(
                                all_types_true, all_types_pred, all_times_true, all_time_preds, all_time_deltas_true,
                                result_save_path, "val", epoch, self.model,seq_scores_val
                            )
                    else:
                        epochs_no_improve += 1
                        print(f'No improvement for {epochs_no_improve} epochs')
                        f.write(f'No improvement for {epochs_no_improve} epochs\n')

                if epochs_no_improve >= patience:
                    with open(result_save_path+'/val.txt', 'a') as f:
                        f.write(f"\nEarly stopping triggered at epoch {epoch}\n")
                    
                    # 🌟 ここで明示的にTest評価を行う（今のモデル状態で計算させる）
                    if dataloader_test:
                        print("Early stopping triggered. Running final test evaluation...")
                        metrics_test, _ = self.run_epoch(dataloader_test, phase='eval', result_save_path=result_save_path)
                        
                        import json
                        with open(result_save_path+'/test.txt', 'a') as f:
                            f.write(f'test metrics of epoch {epoch}: {metrics_test}\n')
                        print("Test metrics at stopping point saved.")
                        
                    break

            if dataloader_test and (save_flag or train_flag):
                # 🌟🌟🌟 ここにお引越し：Valの直前にリセットし、ヘッダーを書く 🌟🌟🌟
                if hasattr(self.model, 'reset_topk_debug_flags'):
                    self.model.reset_topk_debug_flags()
                    self.model.result_save_path = result_save_path  # 👈【追加】モデルに保存先パスを教える
                    try:
                        # 👈【変更】保存先を result_save_path の中にする
                        log_file_path = f"{result_save_path}/topk_debug_log.txt"
                        with open(log_file_path, "a", encoding="utf-8") as f:
                            f.write(f"\n{'='*50}\n 🚀 Epoch {epoch} の テスト Top-k 単語観測 (Validation)\n{'='*50}\n")
                    except Exception:
                        pass
                metrics_test, test_eval_data = self.run_epoch(dataloader_test, phase='eval', result_save_path=result_save_path)
                print(f'test metrics of epoch {epoch}: {metrics_test}')
                with open(result_save_path+'/test.txt', 'a') as f:
                    f.write(f'test metrics of epoch {epoch}: {metrics_test}\n')
                # Only save detailed tessssssssssst analysis if it's the best epoch so far
                if(epochs_no_improve == 0):
                    (all_types_true_test, all_types_pred_test, all_times_true_test, all_time_preds_test, all_time_deltas_true_test, seq_scores_test) = test_eval_data
                    if self.temporal_emb_type in ['MLP_token_renew_vocab_topk_loss_ph','MLP_token_renew_vocab_loss_ph_scaled','MLP_token_renew_loss', 'MLP_token_renew_loss_p', 'MLP_token_renew_loss_h','MLP_token_renew_loss_ph','MLP_token_renew_topk_loss_p', 'MLP_token_renew_topk_loss_h', 'MLP_token_renew_topk_loss_ph','MLP_token_renew_vocab_loss_ph']:
                                
                        quantitative_analysis_func(
                            all_types_true_test, all_types_pred_test, all_times_true_test, all_time_preds_test, all_time_deltas_true_test,
                            result_save_path, f"test", epoch,self.model, seq_scores_test
                        )
            if save_flag or train_flag:
                self.plot_learning_curves(result_save_path)


    # 修正前: def save(self, model_weight_path: str, weight_path: str):
    def save(self, model_weight_path: str, weight_path: str, epoch: int = 0, metrics_val_best: dict = None):
        self.model.save_models(model_weight_path)
        filtered_weights = {
            k: v for k, v in self.model.state_dict().items()
            if not k.startswith("model.") and not k.startswith("lm_head")
        }
        torch.save(filtered_weights, weight_path+'/model.pth')

        # ⬇️ ここから追加: オプティマイザ等の勢いと乱数状態をチェックポイントとして保存
        checkpoint = {
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'metrics_val_best': metrics_val_best,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'np_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate()
        }
        torch.save(checkpoint, weight_path+'/training_checkpoint.pth')

    def load(self, weight_path: str):
        self.model.load_state_dict(torch.load(weight_path+'/model.pth'), strict=False)

    def plot_learning_curves(self, result_save_path: str):
        """ログファイルを読み込んで学習曲線の画像を保存する"""
        print(f"\nSaving learning curves to {result_save_path}...")
        
        # 🌟 変更: Train 用の F1, RMSE, Accuracy のリストを追加
        epochs, train_losses, train_lls, train_f1s, train_rmses, train_accuracies,train_qwk,train_type_loss,train_time_loss,train_nll_loss =[],[],[], [], [], [], [], [], [],[]
        val_epochs, val_losses, val_lls, val_f1s, val_rmses, val_accuracies,val_qwk,val_type_loss,val_time_loss,val_nll_loss= [],[],[], [],[], [], [], [], [],[]
        test_epochs, test_losses, test_lls, test_f1s, test_rmses, test_accuracies,test_qwk,test_type_loss,test_time_loss,test_nll_loss=[],[],[], [], [], [], [], [], [],[]

        train_pattern = re.compile(r"train metrics of epoch (\d+): (\{.*?\})")
        val_pattern = re.compile(r"validation metrics of epoch (\d+): (\{.*?\})")
        test_pattern = re.compile(r"test metrics of epoch (\d+): (\{.*?\})")

        # train.txt の読み込み
        try:
            with open(os.path.join(result_save_path, 'train.txt'), 'r', encoding='utf-8') as f:
                for line in f:
                    match = train_pattern.search(line)
                    if match:
                        epoch = int(match.group(1))
                        metrics = ast.literal_eval(match.group(2))
                        epochs.append(epoch)
                        train_losses.append(metrics.get('train_loss'))
                        train_lls.append(metrics.get('log_likelihood'))
                        train_qwk.append(metrics.get('qwk'))
                        train_type_loss.append(metrics.get('train_type_loss'))
                        train_time_loss.append(metrics.get('train_time_loss'))
                        train_nll_loss.append(metrics.get('train_nll_loss'))
                        # 🌟 ここに追加
                        train_f1s.append(metrics.get('f1'))
                        train_rmses.append(metrics.get('rmse'))
                        train_accuracies.append(metrics.get('accuracy'))
        except FileNotFoundError:
            pass

        # val.txt と test.txt の読み込み部分は変更なし
        try:
            with open(os.path.join(result_save_path, 'val.txt'), 'r', encoding='utf-8') as f:
                for line in f:
                    match = val_pattern.search(line)
                    if match:
                        epoch = int(match.group(1))
                        metrics = ast.literal_eval(match.group(2))
                        val_epochs.append(epoch)
                        val_losses.append(metrics.get('val_loss'))
                        val_lls.append(metrics.get('log_likelihood'))
                        val_f1s.append(metrics.get('f1'))
                        val_qwk.append(metrics.get('qwk'))
                        val_type_loss.append(metrics.get('val_type_loss'))
                        val_time_loss.append(metrics.get('val_time_loss'))
                        val_nll_loss.append(metrics.get('val_nll_loss'))
                        val_rmses.append(metrics.get('rmse'))
                        val_accuracies.append(metrics.get('accuracy'))
        except FileNotFoundError:
            pass

        try:
            with open(os.path.join(result_save_path, 'test.txt'), 'r', encoding='utf-8') as f:
                for line in f:
                    match = test_pattern.search(line)
                    if match:
                        epoch = int(match.group(1))
                        metrics = ast.literal_eval(match.group(2))
                        test_epochs.append(epoch)
                        test_losses.append(metrics.get('val_loss'))
                        test_lls.append(metrics.get('log_likelihood'))
                        test_f1s.append(metrics.get('f1'))
                        test_rmses.append(metrics.get('rmse'))
                        test_type_loss.append(metrics.get('val_type_loss'))
                        test_time_loss.append(metrics.get('val_time_loss'))
                        test_nll_loss.append(metrics.get('val_nll_loss'))
                        test_qwk.append(metrics.get('qwk'))
                        test_accuracies.append(metrics.get('accuracy'))
        except FileNotFoundError:
            pass

        # 1. Loss
        plt.figure(figsize=(8, 5))
        if epochs: plt.plot(epochs, train_losses, label='Train Loss', marker='o', color='tab:blue')
        if val_epochs: plt.plot(val_epochs, val_losses, label='Val Loss', marker='s', color='tab:orange')
        if test_epochs: plt.plot(test_epochs, test_losses, label='Test Loss', marker='x', color='tab:green')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(result_save_path, 'curve_loss.png'), bbox_inches='tight')
        plt.close()

        # 2. Log-Likelihood
        plt.figure(figsize=(8, 5))
        if epochs: plt.plot(epochs, train_lls, label='Train Log-Likelihood', marker='o', color='tab:blue')
        if val_epochs: plt.plot(val_epochs, val_lls, label='Val Log-Likelihood', marker='s', color='tab:orange')
        if test_epochs: plt.plot(test_epochs, test_lls, label='Test Log-Likelihood', marker='x', color='tab:green')
        plt.title('Log-Likelihood (Higher is better)')
        plt.xlabel('Epoch')
        plt.ylabel('Log-Likelihood')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(result_save_path, 'curve_log_likelihood.png'), bbox_inches='tight')
        plt.close()

        # 3. F1 Score (🌟 Train F1 を追加)
        plt.figure(figsize=(8, 5))
        if epochs and any(v is not None for v in train_f1s): plt.plot(epochs, train_f1s, label='Train F1', marker='o', color='tab:blue')
        if val_epochs: plt.plot(val_epochs, val_f1s, label='Validation F1', marker='s', color='tab:orange')
        if test_epochs: plt.plot(test_epochs, test_f1s, label='Test F1', marker='x', color='tab:green')
        plt.title('F1 Score (Event Type)')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(result_save_path, 'curve_f1.png'), bbox_inches='tight')
        plt.close()

        # 3. F1 Score (🌟 Train F1 を追加)
        plt.figure(figsize=(8, 5))
        if epochs and any(v is not None for v in train_qwk): plt.plot(epochs, train_qwk, label='Train F1', marker='o', color='tab:blue')
        if val_epochs: plt.plot(val_epochs, val_qwk, label='Validation qwk', marker='s', color='tab:orange')
        if test_epochs: plt.plot(test_epochs, test_qwk, label='Test qwk', marker='x', color='tab:green')
        plt.title('qwk (Event Type)')
        plt.xlabel('Epoch')
        plt.ylabel('qwk')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(result_save_path, 'curve_qwk.png'), bbox_inches='tight')
        plt.close()

        # 4. RMSE (🌟 Train RMSE を追加)
        plt.figure(figsize=(8, 5))
        if epochs and any(v is not None for v in train_rmses): plt.plot(epochs, train_rmses, label='Train RMSE', marker='o', color='tab:blue')
        if val_epochs: plt.plot(val_epochs, val_rmses, label='Validation RMSE', marker='s', color='tab:orange')
        if test_epochs: plt.plot(test_epochs, test_rmses, label='Test RMSE', marker='x', color='tab:green')
        plt.title('RMSE (Event Time - Lower is better)')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(result_save_path, 'curve_rmse.png'), bbox_inches='tight')
        plt.close()

        # 5. Accuracy (🌟 Train Accuracy を追加)
        plt.figure(figsize=(8, 5))
        if epochs and any(v is not None for v in train_accuracies): plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o', color='tab:blue')
        if val_epochs: plt.plot(val_epochs, val_accuracies, label='Validation Accuracy', marker='s', color='tab:orange')
        if test_epochs: plt.plot(test_epochs, test_accuracies, label='Test Accuracy', marker='x', color='tab:green')
        plt.title('Accuracy (Event Type)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(result_save_path, 'curve_accuracy.png'), bbox_inches='tight')
        plt.close()

        # 5. Accuracy (🌟 Train Accuracy を追加)
        plt.figure(figsize=(8, 5))
        if epochs and any(v is not None for v in train_type_loss): plt.plot(epochs, train_type_loss, label='Train CE loss', marker='o', color='tab:blue')
        if val_epochs: plt.plot(val_epochs, val_type_loss, label='Validation CE loss', marker='s', color='tab:orange')
        if test_epochs: plt.plot(test_epochs, test_type_loss, label='Test CE loss', marker='x', color='tab:green')
        plt.title('CE loss (Event Type)')
        plt.xlabel('Epoch')
        plt.ylabel('CE loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(result_save_path, 'curve_type_loss.png'), bbox_inches='tight')
        plt.close()

        # 5. Accuracy (🌟 Train Accuracy を追加)
        plt.figure(figsize=(8, 5))
        if epochs and any(v is not None for v in train_time_loss): plt.plot(epochs, train_time_loss, label='Train mse loss', marker='o', color='tab:blue')
        if val_epochs: plt.plot(val_epochs, val_time_loss, label='Validation mse loss', marker='s', color='tab:orange')
        if test_epochs: plt.plot(test_epochs, test_time_loss, label='Test mse loss', marker='x', color='tab:green')
        plt.title('mse loss (Event Type)')
        plt.xlabel('Epoch')
        plt.ylabel('mse loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(result_save_path, 'curve_time_loss.png'), bbox_inches='tight')
        plt.close()

        # 5. Accuracy (🌟 Train Accuracy を追加)
        plt.figure(figsize=(8, 5))
        if epochs and any(v is not None for v in train_nll_loss): plt.plot(epochs, train_nll_loss, label='Train ll loss', marker='o', color='tab:blue')
        if val_epochs: plt.plot(val_epochs, val_nll_loss, label='Validation ll loss', marker='s', color='tab:orange')
        if test_epochs: plt.plot(test_epochs, test_nll_loss, label='Test ll loss', marker='x', color='tab:green')
        plt.title('ll loss (Event Type)')
        plt.xlabel('Epoch')
        plt.ylabel('ll loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(result_save_path, 'curve_nll_loss.png'), bbox_inches='tight')
        plt.close()
        
        print(f"✅ Learning curves successfully saved in: {result_save_path}")