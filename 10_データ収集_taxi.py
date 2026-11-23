#!/usr/bin/env python
# -*- coding: utf-8 -*-
# v03.04 (特定パス収集版)
import os
import codecs
import datetime

# 【変更】収集するファイルの相対パスリスト（実行ディレクトリ基準）
TARGET_PATHS = [
    'src/tpp_llm/taxi_renew_vocab2/analysis.py',
    'src/tpp_llm/taxi_renew_vocab2/common_utils.py',
    'src/tpp_llm/taxi_renew_vocab2/data.py',
    'src/tpp_llm/taxi_renew_vocab2/layers.py',
    'src/tpp_llm/taxi_renew_vocab2/model.py',
    'src/tpp_llm/taxi_renew_vocab2/runner.py',
    'src/tpp_llm/taxi_renew_vocab2/utils.py',
    'scripts/taxi_renew_vocab2/train_tpp_llm.py',

]

# ファイル階層を表示する際に除外するディレクトリのリスト
EXCLUDE_DIRS2 = ['configs','file_sh','notebook','images,''analysis','data','save_model','save_weight','result','.git','logs','old','venv','__pycache__', 'venv', 'migrations', 'static', 'mailform']

# 出力ファイル名
OUTPUT_FILE = "./11_開発状況1123_taxi_vocab.txt"

# 大きすぎるファイルの閾値（バイト単位）
MAX_FILE_SIZE = 10
MAX_FILE_SIZE_BYTE = MAX_FILE_SIZE * 1024 * 1024  # 10MB (0.01MB=10KB)
print(f"最大サイズ：{MAX_FILE_SIZE}MB")

# 設定情報の出力
print(f"収集する特定のパスリスト: {TARGET_PATHS}")
print(f"ファイル階層を表示する際に除外するディレクトリのリスト: {EXCLUDE_DIRS2}")

# 最大サイズを超えたファイルを格納するリスト
large_files = []

def collect_files(root_dir):
    collected_files = []
    for rel_path in TARGET_PATHS:
        # OSに合わせてパスのスラッシュを正規化して結合
        file_path = os.path.normpath(os.path.join(root_dir, rel_path))
        
        # ファイルが存在するかチェック
        if os.path.exists(file_path) and os.path.isfile(file_path):
            collected_files.append(file_path)
            
            # ファイルサイズをチェック
            file_size = os.path.getsize(file_path)
            if file_size > MAX_FILE_SIZE_BYTE:
                large_files.append((file_path, file_size / (1024 * 1024)))
        else:
            print(f"[警告] 指定されたファイルが見つかりません: {rel_path}")
            
    return collected_files

def read_file_content(file_path):
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE_BYTE:
        # ファイルサイズが大きすぎる場合はメッセージを返す
        file_extension = os.path.splitext(file_path)[1]
        return f"この{file_extension}ファイルは{MAX_FILE_SIZE}MB以上なので省略。機能追加、修正にあたり中身を見る必要があれば<テキスト概要>に記載の方法で追加の情報提供を依頼してください。。"
    
    try:
        with codecs.open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        return "このファイルはUTF-8でエンコードされていないため読み込めません。"

def display_directory_tree(directory, prefix="", output_file=None):
    """ディレクトリ構造をツリー形式で表示/ファイル出力する"""
    # ディレクトリ内のファイルとフォルダを取得
    entries = os.listdir(directory)
    entries = sorted([e for e in entries if e not in EXCLUDE_DIRS2])
    
    for i, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        is_last = i == len(entries) - 1
        
        # 現在の項目のプレフィックスを作成
        current_prefix = "└── " if is_last else "├── "
        # 次の階層のプレフィックスを作成
        next_prefix = "    " if is_last else "│   "
        
        # ファイル/ディレクトリ行を作成
        line = f"{prefix}{current_prefix}{entry}"
        
        # 出力先に応じて処理
        if output_file:
            output_file.write(line + "\n")
        else:
            print(line)
        
        # ディレクトリの場合は再帰的に処理
        if os.path.isdir(path):
            display_directory_tree(path, prefix + next_prefix, output_file)

def write_overview(output_file):
    """テキストの概要部分を書き込む"""
    overview = f"""<テキスト概要>
このテキストはAIに対して開発状況を共有するものです。この開発状況を見て修正や機能追加、提案などを行ってください。
修正を提案する場合、修正がしやすいようにできるかぎり関数毎、もしくはクラス、小さいファイルなら全文を教えて下さい。全文ではない場合もインデントを元のインデントに合わせてください。
修正がファイル全文の場合は「全文修正」と一言伝えてください。
修正を提案する際、修正のファイルパス提示するようにお願いします。
なるべくデータ量を小さくするため、{MAX_FILE_SIZE}MB以上のデータは省略されています。そのため、もし、さらなる情報が必要な場合は、以下のように伝えてください。その後追加情報を提供します。〇〇.xxは追加情報が必要なファイル名です。複数必要であれば必要に応じてコマンドを修正してください。
「echo "<〇〇.xx>" > 12_追加提供情報データ.txt && cat 〇〇.xx >> 12_追加提供情報データ.txt && echo "</〇〇.xx>" >> 12_追加提供情報データ.txt」

収集する特定のパスリスト: {TARGET_PATHS}
ファイル階層を表示する際に除外するディレクトリのリスト: {EXCLUDE_DIRS2}
</テキスト概要>

"""
    output_file.write(overview)

def main():
    root_directory = os.getcwd()  # 現在のディレクトリを使用
    
    # ファイル収集と内容出力
    collected_files = collect_files(root_directory)
    with codecs.open(OUTPUT_FILE, 'w', encoding='utf-8') as out_file:
        # テキストの概要を出力
        write_overview(out_file)
        
        # ディレクトリ構造を出力
        out_file.write("<ディレクトリ構造>\n")
        display_directory_tree(root_directory, output_file=out_file)
        out_file.write("</ディレクトリ構造>\n\n")
        
        # ファイル内容を出力
        for file_path in collected_files:
            out_file.write(f"<ファイルパス>{file_path}</ファイルパス>\n")
            out_file.write("<ファイル中身>\n")
            out_file.write(read_file_content(file_path))
            out_file.write("\n</ファイル中身>\n\n")
    
    # 最大サイズを超えたファイルの一覧を表示
    if large_files:
        print(f"\n最大サイズ（{MAX_FILE_SIZE}MB）を超えたファイル一覧:")
        for file_path, size_mb in large_files:
            print(f"  - {file_path} ({size_mb:.2f}MB)")
    
    # 出力ファイルのサイズを取得
    output_file_size = os.path.getsize(OUTPUT_FILE)
    output_file_size_kb = output_file_size / 1024
    output_file_size_mb = output_file_size_kb / 1024
    
    print(f"{len(collected_files)}個のファイルが {OUTPUT_FILE} にまとめられました。")
    print(f"保存場所は{OUTPUT_FILE}")
    print(f"出力ファイルのサイズ: {output_file_size_kb:.2f}KB ({output_file_size_mb:.2f}MB)")

if __name__ == "__main__":
    main()