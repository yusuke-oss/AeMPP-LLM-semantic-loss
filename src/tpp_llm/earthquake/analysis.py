"""
TPP-LLM Analysis and Visualization Tools
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

# ====================================================================
# 1. Qualitative & Quantitative Sequence Evaluation
# ====================================================================

def visualize_qualitative_sequence(raw_data, save_dir, phase_name, epoch, num_examples=5):
    """
    Generate trajectory plots for qualitative evaluation.
    Top: True vs Predicted Event Type (Discrete)
    Bottom: Absolute Cumulative Time (Lower bound = 0)
    """
    print(f"Generating qualitative sequence plots (Type & Abs Time) for epoch {epoch}...")
    
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)
    selected_seqs = raw_data[:num_examples]
    
    type_to_y = {0: 2, 1: 1, 2: 0} 
    y_labels = {0: "Small", 1: "Medium", 2: "Large"}

    # Organize neatly into the "plots_sequence" directory
    vis_dir = os.path.join(save_dir, f"analysis/plots_sequence/{phase_name}")
    os.makedirs(vis_dir, exist_ok=True)

    for i, seq in enumerate(selected_seqs):
        true_types = np.array(seq['true_types']).flatten()
        pred_types = np.array(seq['pred_types']).flatten()
        true_times = np.array(seq['true_times']).flatten()
        pred_times = np.array(seq['pred_times']).flatten()

        min_len = min(len(true_types), len(pred_types), len(true_times), len(pred_times))
        true_types = true_types[:min_len]
        pred_types = pred_types[:min_len]
        true_times = true_times[:min_len]
        pred_times = pred_times[:min_len]

        try:
            true_types = true_types.astype(int)
            pred_types = pred_types.astype(int)
        except ValueError:
            pass 

        true_y = [type_to_y.get(t, t) for t in true_types]
        pred_y = [type_to_y.get(t, t) for t in pred_types]
        steps = np.arange(1, min_len + 1)

        acc_str = "N/A"
        if len(true_types) > 0:
            type_acc = np.mean(true_types == pred_types)
            acc_str = f"Type Acc:\n{type_acc:.1%}"

        rmse_str = "N/A"
        if len(true_times) > 0:
            time_rmse = np.sqrt(np.mean((true_times - pred_times) ** 2))
            rmse_str = f"Time RMSE:\n{time_rmse:.4f}"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
        plt.subplots_adjust(right=0.85, hspace=0.15)

        # ==========================================
        # Upper plot: Event Types
        # ==========================================
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax1.grid(True, axis='x', linestyle=':', alpha=0.5)

        ax1.plot(steps, true_y, color='#1f77b4', marker='s', markersize=10, 
                 linestyle='-', linewidth=2, label='True Type', alpha=0.6)
        ax1.plot(steps, pred_y, color='#ff7f0e', marker='o', markersize=8, 
                 linestyle='--', linewidth=2, label='Predicted Type', alpha=0.9)

        yticks_vals = sorted(y_labels.keys())
        ax1.set_yticks(yticks_vals)
        ax1.set_yticklabels([y_labels[v] for v in yticks_vals])
        ax1.set_ylim(-0.5, 2.5)
        
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=12)

        props = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='gray')
        ax1.text(1.02, 0.0, acc_str, transform=ax1.transAxes, fontsize=12, 
                 verticalalignment='bottom', bbox=props)

        # ==========================================
        # Lower plot: Absolute Times
        # ==========================================
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.plot(steps, true_times, color='black', marker='s', markersize=6,
                 linestyle='-', label='True Time', alpha=0.6)
        ax2.plot(steps, pred_times, color='green', marker='^', markersize=6,
                 linestyle='--', label='Predicted Time', alpha=0.8)
        ax2.set_ylim(bottom=0) 
        
        ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=12)

        ax2.text(1.02, 0.0, rmse_str, transform=ax2.transAxes, fontsize=12, 
                 verticalalignment='bottom', bbox=props)

        save_path = os.path.join(vis_dir, f"seq_{i}_type_time.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    print(f"Saved qualitative plots to {vis_dir}")

def perform_quantitative_analysis(
    all_types_true, all_types_pred,
    all_times_true, all_time_preds,
    all_time_deltas_true,
    result_save_path, phase_name,
    epoch, # Passed from the runner
    model=None  ,# 🌟 model を受け取れるように追加,
    seq_scores=None
):
    """
    Evaluate quantitative metrics (Confusion Matrix, RMSE, Classification Report).
    Results are saved neatly in the detailed analysis directory by epoch.
    """
    
    # Create an epoch-specific directory within the detailed metrics folder
    quant_dir = os.path.join(result_save_path, "analysis", f"detail/epoch_{epoch}/{phase_name}")
    os.makedirs(quant_dir, exist_ok=True)

    # 1. Confusion Matrix
    try:
        # 【修正後】（0, 1, 2 の順に合わせる）
        class_names = ['Large', 'Medium', 'Small']
        cm = confusion_matrix(all_types_true, all_types_pred, labels=[0, 1, 2], normalize='true')
        sns.reset_orig()  # Reset seaborn settings to prevent layout issues
        import matplotlib as mpl
        mpl.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
        
        # --- Clean Version (Untitled for Academic Papers) ---
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0.0, vmax=1.0, square=True, cbar_kws={"shrink": .8})
        ax.set_ylabel('True label', fontweight='bold')
        ax.set_xlabel('Predicted label', fontweight='bold')
        
        # Save with a simple filename
        plt.savefig(os.path.join(quant_dir, f'confusion_matrix.png'), 
                    bbox_inches='tight', dpi=150)
        plt.close(fig)
        
    except Exception as e:
        print(f"Failed to generate Confusion Matrix: {e}")

    # 2. Time Error Analysis
    try:
        bins_hours = [0, 1, 6, 12, np.inf]
        bin_labels = ['(0-1 hour)', '(1-6 hours)', '(6-12 hours)', '(12+ hours)']
        
        all_time_deltas_true_hours = all_time_deltas_true * 24.0
        bin_indices = np.digitize(all_time_deltas_true_hours, bins=bins_hours)
        
        analysis_log_time = f"--- {phase_name} Absolute Time RMSE by Intervals ---\n"
        
        for i in range(1, len(bins_hours)):
            label = bin_labels[i-1]
            mask = (bin_indices == i)
            
            if np.sum(mask) == 0:
                analysis_log_time += f"{label}: N=0\n"
                continue
                
            true_abs_times_in_bin = all_times_true[mask]
            pred_abs_times_in_bin = all_time_preds[mask]
            rmse_in_bin = np.sqrt(np.mean((true_abs_times_in_bin - pred_abs_times_in_bin) ** 2))
            
            analysis_log_time += f"{label}: N={np.sum(mask)}, RMSE={rmse_in_bin:.4f}\n"
    except Exception as e:
        analysis_log_time = f"Time Error Analysis Failed: {e}\n"

    # 3. Classification Report
    try:
        report = classification_report(all_types_true, all_types_pred, target_names=class_names, digits=4)
        bal_acc = balanced_accuracy_score(all_types_true, all_types_pred)
        analysis_log_report = f"--- {phase_name} Classification Report ---\n{report}\nBalanced Accuracy: {bal_acc:.4f}\n"
    except Exception as e:
        analysis_log_report = f"Classification Report Failed: {e}\n"
    
    # 4. Save Logs
    try:
        combined_log = analysis_log_time + "\n" + analysis_log_report
        #print(combined_log)
        
        with open(os.path.join(quant_dir, f'analysis_report.txt'), 'w') as f:
            f.write(combined_log)
            
    except Exception as e:
        print(f"Failed to save analysis logs: {e}")

    # 5. Call Sequence Qualitative Analysis
    if seq_scores and 'raw_data' in seq_scores:
        visualize_qualitative_sequence(seq_scores['raw_data'], result_save_path, phase_name, epoch, num_examples=30)
        # 🌟🌟🌟 追加: 新しく作ったCodebook単語可視化関数を呼び出す 🌟🌟🌟
        shared_sequence_indices =visualize_codebook_selection(seq_scores['raw_data'], result_save_path, phase_name, epoch, num_examples=5, max_events=13)
        # --- perform_quantitative_analysis の一番最後に追加 ---
        # 🌟🌟🌟 ここに追加：単語の多様性（全体集計）を実行 🌟🌟🌟
        analyze_vocabulary_diversity(seq_scores['raw_data'], result_save_path, phase_name, epoch)
        # 🌟🌟🌟 追加: t-SNEによる軌跡可視化を実行 🌟🌟🌟
        visualize_codebook_tsne_trajectory(
            seq_scores['raw_data'], result_save_path, phase_name, epoch, model=model, num_examples=len(shared_sequence_indices),max_events=14,selected_indices=shared_sequence_indices
        )
        
        # 🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟
        
def analyze_vocabulary_diversity(raw_data, save_dir, phase_name, epoch):
    """
    全シーケンス・全イベントを通じて、どの単語が何回アンカーとして選ばれたかを
    集計し、頻度分布（ヒストグラム）として保存する。
    """
    print(f"📊 Analyzing vocabulary diversity for {phase_name} (Epoch {epoch})...")
    
    from collections import Counter
    import matplotlib.pyplot as plt
    import os

    # 保存先ディレクトリの作成
    div_dir = os.path.join(save_dir, f"analysis/vocabulary_diversity/{phase_name}")
    os.makedirs(div_dir, exist_ok=True)

    all_words_p = []
    all_words_h = []

    # 全シーケンスの全イベントから、選ばれた単語をフラットなリストに集約
    for seq in raw_data:
        if 'words_p' in seq and seq['words_p']:
            all_words_p.extend(seq['words_p'])
        if 'words_h' in seq and seq['words_h']:
            all_words_h.extend(seq['words_h'])

    # 集計とプロットを行う内部関数
    def plot_and_save_counter(words_list, label_name):
        if not words_list:
            return

        # N/A などの不要な文字を除去し、綺麗にカウント
        cleaned_words = [str(w).replace('Ġ', '').replace(' ', '') for w in words_list if w != "N/A"]
        word_counts = Counter(cleaned_words)
        
        total_events = len(cleaned_words)
        unique_words = len(word_counts)
        
        # 出現回数順にソート
        sorted_counts = word_counts.most_common()

        # --- テキストログとして保存 ---
        log_path = os.path.join(div_dir, f"epoch_{epoch}_{label_name}_diversity_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Vocabulary Diversity Report ({label_name}) ===\n")
            f.write(f"Total Evaluated Events : {total_events}\n")
            f.write(f"Unique Words Selected  : {unique_words} words\n")
            f.write(f"--------------------------------------------------\n")
            f.write(f"{'Rank':<5} | {'Word':<20} | {'Count':<8} | {'Ratio (%)':<8}\n")
            f.write(f"--------------------------------------------------\n")
            for i, (word, count) in enumerate(sorted_counts):
                ratio = (count / total_events) * 100
                f.write(f"{i+1:<5} | {word:<20} | {count:<8} | {ratio:.2f}%\n")

        # --- ヒストグラム（棒グラフ）として保存 ---
        # 上位30単語だけをプロット
        top_k = min(30, unique_words)
        plot_words = [item[0] for item in sorted_counts[:top_k]]
        plot_counts = [item[1] for item in sorted_counts[:top_k]]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(plot_words, plot_counts, color='skyblue', edgecolor='black')
        
        ax.set_title(f"Top {top_k} Selected Words in {label_name} (Epoch {epoch})\nTotal Events: {total_events} | Unique Words: {unique_words}", fontsize=14)
        ax.set_xlabel("Selected Word", fontsize=12)
        ax.set_ylabel("Frequency (Hits)", fontsize=12)
        
        # x軸のラベルを回転させて見やすくする
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        img_path = os.path.join(div_dir, f"epoch_{epoch}_{label_name}_diversity_plot.png")
        plt.savefig(img_path, dpi=150)
        plt.close(fig)

    # Input(P) と Hidden(H) それぞれで実行
    plot_and_save_counter(all_words_p, "Input_P_Layer")
    plot_and_save_counter(all_words_h, "Hidden_H_Layer")
    
    print(f"✅ Vocabulary diversity analysis saved to {div_dir}")

def visualize_codebook_selection(raw_data, save_dir, phase_name, epoch, num_examples=5, max_events=40):
    """
    提案手法で選ばれた単語と、正解・予測のズレを可視化するツール。
    ※ グラフ内に吹き出しを配置しつつ、4段階の高さレーンを使って重なりを完全に防ぎます。
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from itertools import groupby

    # 🌟 修正後: パスに epoch を追加して、エポックごとのフォルダを作成する
    vis_dir = os.path.join(save_dir, f"analysis/plots_codebook/{phase_name}/epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"📊 [Earthquake Viz] Generating plots with anti-overlap dynamic boxes for {phase_name} (Epoch {epoch})...")
    
    # ====================================================
    # 1. 評価・フィルタリング・ソートの処理
    # ====================================================
    scored_seqs = []
    for i, seq in enumerate(raw_data):
        if not seq.get('words_p') and not seq.get('words_h'):
            continue
            
        if len(seq['true_times']) < 8:
            continue

        n = min(len(seq['true_times']), max_events)
        types_true = seq['true_types'][:n]
        types_pred = seq['pred_types'][:n]
        times_true = np.array(seq['true_times'][:n])
        times_pred = np.array(seq['pred_times'][:n])
        
        if len(set(types_true)) != 2:
            continue
            
        acc = np.mean(np.array(types_true) == np.array(types_pred))
        rmse = np.sqrt(np.mean((times_true - times_pred)**2))
        
        scored_seqs.append({
            'original_idx': i, 'seq': seq, 'acc': acc, 'rmse': rmse
        })
        
    scored_seqs.sort(key=lambda x: ( x['rmse']))
    top_seqs = scored_seqs[:num_examples]
    
    # ====================================================
    # 2. 描画処理（上位の系列のみ）
    # ====================================================
    plotted_count = 0
    for item in top_seqs:
        orig_i = item['original_idx']
        seq = item['seq']
        acc = item['acc']
        rmse = item['rmse']
        
        plotted_count += 1
        n = min(len(seq['true_times']), max_events)
        
        times_true = seq['true_times'][:n]
        types_true = seq['true_types'][:n]  
        plot_y_true = [2 - t for t in types_true]
        
        times_pred = seq['pred_times'][:n]
        types_pred = seq['pred_types'][:n]  
        plot_y_pred = [2 - t for t in types_pred]

        # 地震用の補助情報
        mags = seq.get('mags', [0]*(n+1))[:n+1]
        deps = seq.get('deps', [0]*(n+1))[:n+1]
        lats = seq.get('lats', [0]*(n+1))[:n+1]
        lons = seq.get('lons', [0]*(n+1))[:n+1]
        time = seq.get('time', [0]*(n+1))[:n+1]
        type = seq.get('type', [0]*(n+1))[:n+1]
        plot_y_type = [2 - t for t in type]
        
        words_p = seq.get('words_p', [])
        words_h = seq.get('words_h', [])

        # 🌟 変更: 縦幅を広げ、上下に吹き出しを逃がせるスペースを確保
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.grid(True, linestyle='--', alpha=0.5)

        # 軌跡（パス）を描画（真値の軌跡を黒・ダークグレーに統一）
        ax.plot(time, plot_y_type, color='#333333', linestyle='-', linewidth=2, alpha=0.6, zorder=1)
        ax.plot(times_pred, plot_y_pred, color='#d62728', linestyle='--', linewidth=2, alpha=0.4, zorder=1)
            # 正解と予測のプロット
            # 正解と予測のプロット（真値を t-SNE と同じ「中が白の×」に統一）
        ax.scatter(
            time, plot_y_type, 
            s=150, 
            marker='X',           # t-SNEと同じ「×」に設定
            facecolors='white',   # 中身を白抜きに
            edgecolors='black',   # 枠線を黒に
            linewidths=1.2,       # 枠線の太さ
            label='True event', 
            zorder=4
        )
        
        ax.scatter(
            times_pred, plot_y_pred, 
            color='#d62728', 
            s=100, 
            marker='s',           # 予測値は区別しやすいように四角（'s'）や別の形にするのもアリ
            label='Predicted event', 
            zorder=4
        )
        for j in range(n+1):
            step_idx = j + 1
            # 3. 単語の整形
            wp = str(words_p[j]).replace('Ġ', '').replace(' ', '') if j < len(words_p) else "N/A"
            wh = str(words_h[j]).replace('Ġ', '').replace(' ', '') if j < len(words_h) else "N/A"
            
            annotation_text_true = (
                rf"$\mathbf{{Event\ {step_idx-1}}}$" + "\n"
                f"Mag.: {mags[j]:.1f}, Dep.: {deps[j]:.1f} km\n"
                f"Lat: {lats[j]:.1f}°, Lon: {lons[j]:.1f}°"
            )

            # ========================================================
            # 🌟 修正: 上下完全分離レイアウト
            # ========================================================
            lane = j % 4  # レーンを5段階に増やして横の被りもさらに減らす
            
            # 2. 正解（青）はクラスに関係なく常にグラフの「上部」に絶対座標で配置
            box_y_true = 2.4 + (lane * 1.1) 
            
            # ----------------------------------------------------
            # 1. 正解（青）の線だけを描画 (zorder=4)
            # ----------------------------------------------------
            ax.annotate(
                "", 
                xy=(time[j], plot_y_type[j]),                    
                xytext=(time[j], box_y_true), 
                arrowprops=dict(arrowstyle="-", color="#333333", lw=0.8, alpha=0.7), 
                zorder=4
            )
            
            # ----------------------------------------------------
            # 2. 正解（青）の吹き出しだけを配置 (zorder=6)
            # ----------------------------------------------------
            ax.text(
                time[j], box_y_true,
                annotation_text_true,
                ha='center', va='bottom', fontsize=12, color='#333333',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#333333", alpha=1.0, lw=0.8), 
                zorder=6
            )

        for j in range(n):
            step_idx = j + 1
            # 1. 予測(根本) から 正解(先端) への矢印
            ax.annotate(
                "", xy=(times_true[j], plot_y_true[j]), xytext=(times_pred[j], plot_y_pred[j]),  
                arrowprops=dict(arrowstyle="->", color="gray", linestyle="--", alpha=0.8, connectionstyle="arc3,rad=0.2", lw=1.5),
                zorder=2
            )
            
            # ========================================================
            # 🌟 修正: 上下完全分離レイアウト
            # ========================================================
            lane = j % 4  # レーンを5段階に増やして横の被りもさらに減らす
            # 4. 統合されたテキスト
            annotation_text = (
                rf"$\mathbf{{Event\ {step_idx}}}$"
            )
            
            # 5. 予測（赤）はクラスに関係なく常にグラフの「下部」に絶対座標で配置
            box_y_pred = -0.3 - (lane * 0.4)
            
            # ----------------------------------------------------
            # 1. まず「線だけ」を描画する（zorder=3で優先度を低くする）
            # ----------------------------------------------------
            ax.annotate(
                "", # テキストは空にする
                xy=(times_pred[j], plot_y_pred[j]), 
                xytext=(times_pred[j], box_y_pred), 
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.0, alpha=0.5), 
                zorder=3  # 🌟 軌跡の線(1,2)よりは上、吹き出しよりは下に設定
            )
            
            # ----------------------------------------------------
            # 2. その上に「吹き出しだけ」を配置する（zorder=5で優先度を高くし、不透明にする）
            # ----------------------------------------------------
            ax.text(
                times_pred[j], box_y_pred, # xytextと同じ座標に置く
                annotation_text,
                ha='center', va='top', fontsize=12, color='#333333',
                bbox=dict(boxstyle="round,pad=0.4", fc="#fffafa", ec="#d62728", alpha=1.0, lw=1.2),
                zorder=5  # 🌟 線よりも高く設定
            )

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Small", "Medium", "Large"], fontsize=15)
        
        # 🌟 変更: 上下の吹き出しが収まるようにY軸の表示範囲をさらに拡張
        ax.set_ylim(-2.0, 7.0) 
        
        # 凡例と注釈を「グラフの枠外（右側）」に配置
        ax.legend(loc='upper right', bbox_to_anchor=(0.0, 1.0), fontsize=15, framealpha=0.9)
        
        
        ax.set_xlabel("Time [day]", fontsize=25)
        ax.set_ylabel("Event type", fontsize=25)

        save_path = os.path.join(vis_dir, f"earthquake_long_{plotted_count:02d}_seq{orig_i}.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    if plotted_count == 0:
        print(f"⚠️ [Earthquake Viz] 条件を満たす系列が見つかりませんでした。")
        return []

    else:
        print(f"✅ [Earthquake Viz] Saved {plotted_count} anti-overlap plots to {vis_dir}")

        # t-SNEでも同じ系列を使うため、元のraw_data indexを返す
        return [item['original_idx'] for item in top_seqs]

    
def visualize_codebook_tsne_trajectory(
    raw_data,
    save_dir,
    phase_name,
    epoch,
    model=None,
    codebook_embs=None,
    codebook_words=None,
    num_examples=5,
    max_events=30,
    use_hidden_codebook=False,
    perplexity=15,
    annot_top_words=100,
    selected_indices=None,

):
    """
    Saved raw_data['input_vector'] / raw_data['hidden_vector'] を直接使用して
    t-SNE による Event 軌跡と Codebook トークンの可視化を行う。
    """

    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import traceback

    from sklearn.manifold import TSNE

    # ============================================================
    # 保存先
    # ============================================================
    vis_dir = os.path.join(
        save_dir,
        f"analysis/plots_tsne/{phase_name}/epoch_{epoch}"
    )

    os.makedirs(
        vis_dir,
        exist_ok=True
    )

    print(
        "\n============================================================"
    )
    print(
        "🎨 [Embedding Visualization (t-SNE Only)]"
    )
    print(
        f"Phase : {phase_name}"
    )
    print(
        f"Epoch : {epoch}"
    )
    print(
        "Mode  : t-SNE"
    )
    print(
        "============================================================"
    )

    # ============================================================
    # DDP 対応
    # ============================================================
    base_model = (
        model.module
        if (
            model is not None
            and hasattr(model, "module")
        )
        else model
    )

    # ============================================================
    # Codebook P / H
    # ============================================================
    codebook_p = None
    codebook_h = None

    codebook_words_p = None
    codebook_words_h = None

    # ============================================================
    # model から codebook を取得
    # ============================================================
    if base_model is not None:

        if (
            hasattr(base_model, "codebook_p")
            and base_model.codebook_p is not None
        ):
            codebook_p = (
                base_model
                .codebook_p
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        if (
            hasattr(base_model, "codebook_h")
            and base_model.codebook_h is not None
        ):
            codebook_h = (
                base_model
                .codebook_h
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        codebook_words_p = getattr(
            base_model,
            "codebook_words_p",
            None
        )
        codebook_words_h = getattr(
            base_model,
            "codebook_words_h",
            None
        )

        generic_words = getattr(
            base_model,
            "codebook_words",
            []
        )

        if codebook_words_p is None:
            codebook_words_p = generic_words

        if codebook_words_h is None:
            codebook_words_h = generic_words

        if codebook_p is None:
            if (
                hasattr(base_model, "codebook")
                and base_model.codebook is not None
            ):
                codebook_p = (
                    base_model
                    .codebook
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

        if codebook_h is None:
            if (
                hasattr(base_model, "codebook")
                and base_model.codebook is not None
            ):
                codebook_h = (
                    base_model
                    .codebook
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

    # ============================================================
    # 明示的 codebook_embs
    # ============================================================
    if codebook_embs is not None:
        if isinstance(
            codebook_embs,
            torch.Tensor
        ):
            codebook_embs = (
                codebook_embs
                .detach()
                .cpu()
                .numpy()
            )

        codebook_embs = np.asarray(
            codebook_embs,
            dtype=np.float32
        )

        if use_hidden_codebook:
            codebook_h = codebook_embs
        else:
            codebook_p = codebook_embs

    # ============================================================
    # 明示的 codebook_words
    # ============================================================
    if codebook_words is not None:
        if use_hidden_codebook:
            codebook_words_h = codebook_words
        else:
            codebook_words_p = codebook_words

    # ============================================================
    # selected_indices
    # ============================================================
    if selected_indices is None:
        print(
            "⚠️ [Embedding Visualization] "
            "selected_indices が指定されていません。"
        )
        return []

    print(
        f"🔗 [Embedding Visualization] "
        f"selected_indices = {selected_indices}"
    )

    # ============================================================
    # mode
    # ============================================================
    modes = [
        {
            "layer_name": "Input (P)",
            "suffix": "input",
            "vector_key": "input_vector",
            "codebook": codebook_p,
            "codebook_words": codebook_words_p,
        },
        {
            "layer_name": "Hidden (H)",
            "suffix": "hidden",
            "vector_key": "hidden_vector",
            "codebook": codebook_h,
            "codebook_words": codebook_words_h,
        }
    ]

    plotted_results = []

    # ============================================================
    # Sequence Loop
    # ============================================================
    for orig_idx in selected_indices:

        if (
            orig_idx < 0
            or orig_idx >= len(raw_data)
        ):
            print(f"⚠️ Invalid sequence index: {orig_idx}")
            continue

        seq = raw_data[orig_idx]

        raw_times = list(seq.get("time", seq.get("tsne_time", [])))
        raw_types = list(seq.get("type", seq.get("tsne_type", [])))

        words_p = seq.get("words_p", [])
        words_h = seq.get("words_h", [])

        # ========================================================
        # Input / Hidden
        # ========================================================
        for mode in modes:

            layer_name = mode["layer_name"]
            suffix = mode["suffix"]
            vector_key = mode["vector_key"]
            cb_embs = mode["codebook"]
            cb_words = mode["codebook_words"]

            tsne_png = None
            tsne_pdf = None

            if cb_embs is None:
                print(f"⚠️ Seq {orig_idx} | {layer_name}: Codebook がありません。")
                continue

            cb_embs = np.asarray(cb_embs, dtype=np.float32)

            if cb_embs.ndim != 2:
                print(f"⚠️ Seq {orig_idx} | {layer_name}: Codebook shape={cb_embs.shape}")
                continue

            # ====================================================
            # ★ Input (P) のときだけ、重複する単語・ベクトルをユニークにする
            # ====================================================
            if suffix == "input" and cb_words is not None:
                seen_words = set()
                unique_indices = []
                for idx, w in enumerate(cb_words):
                    w_clean = str(w).replace("Ġ", "").replace(" ", "")
                    if w_clean not in seen_words:
                        seen_words.add(w_clean)
                        unique_indices.append(idx)
                
                cb_embs = cb_embs[unique_indices]
                cb_words = [cb_words[i] for i in unique_indices]

            saved_vectors = seq.get(vector_key, None)

            if saved_vectors is None:
                print(f"⚠️ Seq {orig_idx} | {layer_name}: raw_data['{vector_key}'] がありません。")
                continue

            try:
                event_embs = np.asarray(saved_vectors, dtype=np.float32)
            except Exception as e:
                print(f"❌ Seq {orig_idx} | {layer_name}: {vector_key} conversion error: {e}")
                continue

            if event_embs.ndim == 1:
                event_embs = event_embs.reshape(1, -1)

            if event_embs.ndim != 2:
                print(f"⚠️ Seq {orig_idx} | {layer_name}: vector shape={event_embs.shape}")
                continue

            if event_embs.shape[1] != cb_embs.shape[1]:
                print(
                    f"⚠️ Seq {orig_idx} | {layer_name}: "
                    f"dimension mismatch: event={event_embs.shape[1]}, codebook={cb_embs.shape[1]}"
                )
                continue

            event_lengths = [len(event_embs), len(raw_times), len(raw_types)]
            if suffix == "input":
                if len(words_p) > 0:
                    event_lengths.append(len(words_p))
            else:
                if len(words_h) > 0:
                    event_lengths.append(len(words_h))

            raw_event_count = min(event_lengths)
            n_events = min(raw_event_count, max_events)

            if n_events <= 0:
                print(f"⚠️ Seq {orig_idx} | {layer_name}: Event がありません。")
                continue

            ev_embs = event_embs[:n_events]

            # ====================================================
            # t-SNE 処理
            # ====================================================
            combined_embs = np.vstack([cb_embs, ev_embs])
            num_codebook_tokens = cb_embs.shape[0]
            total_samples = combined_embs.shape[0]

            if total_samples > 2:
                effective_perplexity = max(
                    1,
                    min(perplexity, total_samples - 1)
                )

                try:
                    tsne = TSNE(
                        n_components=2,
                        perplexity=effective_perplexity,
                        random_state=42,
                        init="pca",
                        learning_rate="auto"
                    )

                    embedded_2d = tsne.fit_transform(combined_embs)

                    cb_2d = embedded_2d[:num_codebook_tokens]
                    ev_2d = embedded_2d[num_codebook_tokens:]

                    fig, ax = plt.subplots(figsize=(14, 10))
                    ax.grid(True, linestyle="--", alpha=0.4)

                    # Codebook scatter
                    ax.scatter(
                        cb_2d[:, 0],
                        cb_2d[:, 1],
                        c="#a0c4ff",
                        alpha=0.45,
                        s=60,
                        edgecolors="#4361ee",
                        label="Codebook Tokens",
                        zorder=2
                    )

                    # ====================================================
                    # Codebook words annotations
                    # Input (P): 従来どおり単語名を表示
                    # Hidden (H): 重複単語に番号を付けて区別
                    # ====================================================
                    if (
                        cb_words is not None
                        and len(cb_words) == num_codebook_tokens
                    ):
                        display_words = []
                        word_counts = {}

                        for w in cb_words:
                            w_clean = str(w).replace("Ġ", "").replace(" ", "")

                            if suffix == "hidden":
                                word_counts[w_clean] = word_counts.get(w_clean, 0) + 1
                                count = word_counts[w_clean]

                                if count == 1:
                                    display_word = w_clean
                                else:
                                    display_word = f"{w_clean} ({count})"
                            else:
                                display_word = w_clean

                            display_words.append(display_word)

                        for idx, word_label in enumerate(display_words):
                            x_pos = cb_2d[idx, 0]
                            y_pos = cb_2d[idx, 1]

                            ax.annotate(
                                word_label,
                                xy=(x_pos, y_pos),
                                xytext=(x_pos + 0.3, y_pos + 0.3),
                                fontsize=8,
                                color="#1d3557",
                                weight="bold",
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    fc="white",
                                    ec="#a0c4ff",
                                    alpha=0.85,
                                    lw=0.6
                                ),
                                arrowprops=dict(
                                    arrowstyle="-",
                                    color="#a0c4ff",
                                    lw=0.8,
                                    shrinkA=0,
                                    shrinkB=4
                                ),
                                zorder=3
                            )


                    try:
                        cmap = plt.colormaps["plasma"]
                    except AttributeError:
                        cmap = plt.cm.get_cmap("plasma")

                    # Event trajectory line
                    if n_events >= 2:
                        ax.plot(
                            ev_2d[:, 0],
                            ev_2d[:, 1],
                            color="#333333",
                            linestyle="--",
                            linewidth=1.8,
                            alpha=0.7,
                            zorder=4,
                            label="Event Trajectory"
                        )

                    # Direction arrows
                    for j in range(n_events - 1):
                        ax.annotate(
                            "",
                            xy=(ev_2d[j + 1, 0], ev_2d[j + 1, 1]),
                            xytext=(ev_2d[j, 0], ev_2d[j, 1]),
                            arrowprops=dict(
                                arrowstyle="->",
                                color="black",
                                lw=1.5,
                                alpha=0.8,
                                mutation_scale=12
                            ),
                            zorder=4
                        )

                    # Event points / IDs
                    for j in range(n_events):
                        event_id = f"E{j:d}"

                        ax.scatter(
                            ev_2d[j, 0], ev_2d[j, 1],
                            facecolors="none", s=180, marker="X",
                            edgecolors="black", lw=0.8, zorder=5
                        )
                            

                        ax.text(
                            ev_2d[j, 0] + 0.3,
                            ev_2d[j, 1] + 0.3,
                            event_id,
                            fontsize=8,
                            fontweight="bold",
                            color="#2b2d42",
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                fc="#fff3bf",
                                ec="#f4a261",
                                alpha=0.9,
                                lw=0.8
                            ),
                            zorder=7
                        )

                    #ax.set_title(
                    #    f"Event Embedding Trajectory (t-SNE)\n"
                    #    f"Seq {orig_idx}, Epoch {epoch} | Phase: {phase_name} | Layer: {layer_name} | Events: {n_events}",
                    #    fontsize=13,
                    #    fontweight="bold"
                    #)
                    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
                    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
                    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=10)

                    # --- 枠線・グリッドを消す設定 ---
                    #plt.grid(False)  # 方眼紙状のグリッド線を消す

                    # もし背景色（グレー）自体を真っ白にしたい場合
                    #ax = plt.gca()
                    #ax.set_facecolor("white")  # 背景色を白に設定
                    # 軸（目盛りと数字）を丸ごと非表示にする
                    #plt.axis("off")

                    tsne_png = os.path.join(vis_dir, f"tsne_trajectory_seq{orig_idx}_{suffix}.png")
                    tsne_pdf = os.path.join(vis_dir, f"tsne_trajectory_seq{orig_idx}_{suffix}.pdf")

                    plt.savefig(tsne_png, bbox_inches="tight", dpi=150)
                    plt.savefig(tsne_pdf, bbox_inches="tight")
                    plt.close(fig)

                    print(f"✅ [t-SNE] Seq {orig_idx} | {layer_name} | Events={n_events}")

                except Exception as e:
                    print(f"❌ [t-SNE] Seq {orig_idx} | {layer_name}: {e}")
                    traceback.print_exc()
            else:
                print(f"⚠️ [t-SNE] Seq {orig_idx} | {layer_name}: sample数不足のためスキップ。")

            # Result 記録 (チェックツールが求めるキーに対応)
            res_dict = {
                "seq_idx": orig_idx,
                "mode": suffix,
                "event_count": n_events,
                "vector_key": vector_key,
                "tsne_png": tsne_png,
                "tsne_pdf": tsne_pdf,
            }
            if suffix == "input":
                res_dict["input_tsne"] = tsne_png if (tsne_png and os.path.exists(tsne_png)) else False
            else:
                res_dict["hidden_tsne"] = tsne_png if (tsne_png and os.path.exists(tsne_png)) else False

            plotted_results.append(res_dict)

    # ============================================================
    # 終了
    # ============================================================
    print(
        "\n============================================================"
    )
    print(
        f"✅ [Embedding Visualization] Saved {len(plotted_results)} results (t-SNE only)"
    )
    print(
        "============================================================"
    )

    return plotted_results
