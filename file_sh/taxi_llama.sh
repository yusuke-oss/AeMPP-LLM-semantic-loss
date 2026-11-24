#(1)AeMPP

# 乱数シードの設定
SEEDS=(42)
llm='llama'

WEIGHT_P_LIST=(
  0
  0.001

)

WEIGHT_H_LIST=(
  0
  0.001
  
)

# ★ 実験セットの定義（1要素）
DATASETS=( "nyc_taxi" ) 
SAVE_NAMES=("20261123/nyc_taxi" ) 
SCRIPT_DIRS=("taxi" )

# --- 共通設定 ---
TEMP_EMB_TYPES=('MLP_token_renew_loss_ph'  )
FLOSEN_FLAG_SET=( true )
USE_TOKENS_SET=( false )
NATURAL_SEP_SET=( false )
EMB_FLOZEN_SET=( false )

# =========================================================
#   Execution Loop
# =========================================================

for seed in "${SEEDS[@]}"; do

  echo "========================================="
  echo " STARTING EXPERIMENTS FOR SEED: $seed "
  echo "========================================="

  for i in "${!TEMP_EMB_TYPES[@]}"; do
    
    DATASET_NAME="${DATASETS[$i]}"
    SAVE_NAME="${SAVE_NAMES[$i]}"
    SCRIPT_DIR="${SCRIPT_DIRS[$i]}"
    TEMP_EMB_TYPE="${TEMP_EMB_TYPES[$i]}"
    
    # フラグの取り出し
    USE_TOKENS="${USE_TOKENS_SET[$i]}"
    NAT_SEP="${NATURAL_SEP_SET[$i]}"
    FLOZEN_PRE="${FLOSEN_FLAG_SET[$i]}"
    EMB_FLOZEN_PRE="${EMB_FLOZEN_SET[$i]}"
    
    # --- パス設定（共通部分） ---
    DATASET_PATH="data/${DATASET_NAME}"
    BASE_RESULT="result/${SAVE_NAME}"
    BASE_WEIGHT_SAVE="save_weight/${SAVE_NAME}"
    BASE_MODEL_LOAD="save_model/${SAVE_NAME}"

    for idx in "${!WEIGHT_P_LIST[@]}"; do
      codebook_weight_p="${WEIGHT_P_LIST[$idx]}"
      codebook_weight_h="${WEIGHT_H_LIST[$idx]}"
      
      # フォルダ名が被らないよう、重みの値もディレクトリ名に追加
      FLAG_STR="${llm}/tokens_${USE_TOKENS}_nat_${NAT_SEP}_flosen_${FLOZEN_PRE}_embflozen_${EMB_FLOZEN_PRE}_wp_${codebook_weight_p}_wh_${codebook_weight_h}"

      SUB_DIR="${TEMP_EMB_TYPE}/${FLAG_STR}/seed_${seed}"

      CUR_RESULT="${BASE_RESULT}/${SUB_DIR}"
      CUR_WEIGHT="${BASE_WEIGHT_SAVE}/${SUB_DIR}"
      CUR_MODEL_LOAD="${BASE_MODEL_LOAD}/${SUB_DIR}"
      
      mkdir -p "$CUR_RESULT" "$CUR_WEIGHT" "$CUR_MODEL_LOAD"

      # --- コマンド引数の構築 ---
      TOKENS_CMD=""
      if [ "$USE_TOKENS" = true ]; then
          TOKENS_CMD="--use_tokens"
          if [ "$NAT_SEP" = true ]; then
              TOKENS_CMD="$TOKENS_CMD --use_natural_separators"
          fi
      fi
      
      if [ "$FLOZEN_PRE" = true ]; then
          FLOZEN_CMD="--flozen_flag --peft_type None"
      else
          FLOZEN_CMD="--peft_type lora"
      fi

      EMB_FLOZEN_CMD=""
      if [ "$EMB_FLOZEN_PRE" = true ]; then
          EMB_FLOZEN_CMD="--emb_flozen_flag"
      fi

    

      echo "-------------------------------------------------------"
      echo " Running Experiment [Index: $idx]"
      echo " Dataset    : $DATASET_NAME"
      echo " Emb Type   : $TEMP_EMB_TYPE"
      echo " Weight P   : $codebook_weight_p"
      echo " Weight H   : $codebook_weight_h"
      echo " Out Dir    : $CUR_RESULT"
      echo "-------------------------------------------------------"

      python "scripts/${SCRIPT_DIR}/train_tpp_llm.py" \
        @configs/tpp_llm_nt.config \
        --dataset_path="${DATASET_PATH}" \
        --result_save_path="${CUR_RESULT}" \
        --model_weight_path="${CUR_MODEL_LOAD}" \
        --weight_path="${CUR_WEIGHT}" \
        --seed="${seed}" \
        --temporal_emb_type="${TEMP_EMB_TYPE}" \
        --device=cuda:1 \
        --codebook_loss_weight_p="${codebook_weight_p}" \
        --codebook_loss_weight_h="${codebook_weight_h}" \
        $TOKENS_CMD $FLOZEN_CMD $EMB_FLOZEN_CMD 
        
        
    done
        
  done
done



#(2)TPP-LLM

# 乱数シードの設定
SEEDS=(42)
llm='llama'

WEIGHT_P_LIST=(
  0
)

WEIGHT_H_LIST=(
  0
)

# ★ 実験セットの定義（1要素）
DATASETS=( "nyc_taxi" ) 
SAVE_NAMES=("20261123/nyc_taxi" ) 
SCRIPT_DIRS=("taxi" )

# --- 共通設定 ---
TEMP_EMB_TYPES=('main')
FLOSEN_FLAG_SET=( false )
USE_TOKENS_SET=( false )
NATURAL_SEP_SET=( false )
EMB_FLOZEN_SET=( false )

# =========================================================
#   Execution Loop
# =========================================================

for seed in "${SEEDS[@]}"; do

  echo "========================================="
  echo " STARTING EXPERIMENTS FOR SEED: $seed "
  echo "========================================="

  for i in "${!TEMP_EMB_TYPES[@]}"; do
    
    DATASET_NAME="${DATASETS[$i]}"
    SAVE_NAME="${SAVE_NAMES[$i]}"
    SCRIPT_DIR="${SCRIPT_DIRS[$i]}"
    TEMP_EMB_TYPE="${TEMP_EMB_TYPES[$i]}"
    
    # フラグの取り出し
    USE_TOKENS="${USE_TOKENS_SET[$i]}"
    NAT_SEP="${NATURAL_SEP_SET[$i]}"
    FLOZEN_PRE="${FLOSEN_FLAG_SET[$i]}"
    EMB_FLOZEN_PRE="${EMB_FLOZEN_SET[$i]}"
    
    # --- パス設定（共通部分） ---
    DATASET_PATH="data/${DATASET_NAME}"
    BASE_RESULT="result/${SAVE_NAME}"
    BASE_WEIGHT_SAVE="save_weight/${SAVE_NAME}"
    BASE_MODEL_LOAD="save_model/${SAVE_NAME}"

    for idx in "${!WEIGHT_P_LIST[@]}"; do
      codebook_weight_p="${WEIGHT_P_LIST[$idx]}"
      codebook_weight_h="${WEIGHT_H_LIST[$idx]}"
      
      # フォルダ名が被らないよう、重みの値もディレクトリ名に追加
      FLAG_STR="${llm}/tokens_${USE_TOKENS}_nat_${NAT_SEP}_flosen_${FLOZEN_PRE}_embflozen_${EMB_FLOZEN_PRE}_wp_${codebook_weight_p}_wh_${codebook_weight_h}"

      SUB_DIR="${TEMP_EMB_TYPE}/${FLAG_STR}/seed_${seed}"

      CUR_RESULT="${BASE_RESULT}/${SUB_DIR}"
      CUR_WEIGHT="${BASE_WEIGHT_SAVE}/${SUB_DIR}"
      CUR_MODEL_LOAD="${BASE_MODEL_LOAD}/${SUB_DIR}"
      
      mkdir -p "$CUR_RESULT" "$CUR_WEIGHT" "$CUR_MODEL_LOAD"

      # --- コマンド引数の構築 ---
      TOKENS_CMD=""
      if [ "$USE_TOKENS" = true ]; then
          TOKENS_CMD="--use_tokens"
          if [ "$NAT_SEP" = true ]; then
              TOKENS_CMD="$TOKENS_CMD --use_natural_separators"
          fi
      fi
      
      if [ "$FLOZEN_PRE" = true ]; then
          FLOZEN_CMD="--flozen_flag --peft_type None"
      else
          FLOZEN_CMD="--peft_type lora"
      fi

      EMB_FLOZEN_CMD=""
      if [ "$EMB_FLOZEN_PRE" = true ]; then
          EMB_FLOZEN_CMD="--emb_flozen_flag"
      fi

    

      echo "-------------------------------------------------------"
      echo " Running Experiment [Index: $idx]"
      echo " Dataset    : $DATASET_NAME"
      echo " Emb Type   : $TEMP_EMB_TYPE"
      echo " Weight P   : $codebook_weight_p"
      echo " Weight H   : $codebook_weight_h"
      echo " Out Dir    : $CUR_RESULT"
      echo "-------------------------------------------------------"

      python "scripts/${SCRIPT_DIR}/train_tpp_llm.py" \
        @configs/tpp_llm_nt.config \
        --dataset_path="${DATASET_PATH}" \
        --result_save_path="${CUR_RESULT}" \
        --model_weight_path="${CUR_MODEL_LOAD}" \
        --weight_path="${CUR_WEIGHT}" \
        --seed="${seed}" \
        --temporal_emb_type="${TEMP_EMB_TYPE}" \
        --device=cuda:1 \
        --codebook_loss_weight_p="${codebook_weight_p}" \
        --codebook_loss_weight_h="${codebook_weight_h}" \
        $TOKENS_CMD $FLOZEN_CMD $EMB_FLOZEN_CMD 
        
        
    done
        
  done
done

#(2)TPP-LLM+NL

# 乱数シードの設定
SEEDS=(42)
llm='llama'

WEIGHT_P_LIST=(
  0
)

WEIGHT_H_LIST=(
  0
)

# ★ 実験セットの定義（1要素）
DATASETS=( "nyc_taxi" ) 
SAVE_NAMES=("20261123/nyc_taxi" ) 
SCRIPT_DIRS=("taxi" )

# --- 共通設定 ---
TEMP_EMB_TYPES=('positional'  )
FLOSEN_FLAG_SET=( false )
USE_TOKENS_SET=( true )
NATURAL_SEP_SET=( true )
EMB_FLOZEN_SET=( false )

# =========================================================
#   Execution Loop
# =========================================================

for seed in "${SEEDS[@]}"; do

  echo "========================================="
  echo " STARTING EXPERIMENTS FOR SEED: $seed "
  echo "========================================="

  for i in "${!TEMP_EMB_TYPES[@]}"; do
    
    DATASET_NAME="${DATASETS[$i]}"
    SAVE_NAME="${SAVE_NAMES[$i]}"
    SCRIPT_DIR="${SCRIPT_DIRS[$i]}"
    TEMP_EMB_TYPE="${TEMP_EMB_TYPES[$i]}"
    
    # フラグの取り出し
    USE_TOKENS="${USE_TOKENS_SET[$i]}"
    NAT_SEP="${NATURAL_SEP_SET[$i]}"
    FLOZEN_PRE="${FLOSEN_FLAG_SET[$i]}"
    EMB_FLOZEN_PRE="${EMB_FLOZEN_SET[$i]}"
    
    # --- パス設定（共通部分） ---
    DATASET_PATH="data/${DATASET_NAME}"
    BASE_RESULT="result/${SAVE_NAME}"
    BASE_WEIGHT_SAVE="save_weight/${SAVE_NAME}"
    BASE_MODEL_LOAD="save_model/${SAVE_NAME}"

    for idx in "${!WEIGHT_P_LIST[@]}"; do
      codebook_weight_p="${WEIGHT_P_LIST[$idx]}"
      codebook_weight_h="${WEIGHT_H_LIST[$idx]}"
      
      # フォルダ名が被らないよう、重みの値もディレクトリ名に追加
      FLAG_STR="${llm}/tokens_${USE_TOKENS}_nat_${NAT_SEP}_flosen_${FLOZEN_PRE}_embflozen_${EMB_FLOZEN_PRE}_wp_${codebook_weight_p}_wh_${codebook_weight_h}"

      SUB_DIR="${TEMP_EMB_TYPE}/${FLAG_STR}/seed_${seed}"

      CUR_RESULT="${BASE_RESULT}/${SUB_DIR}"
      CUR_WEIGHT="${BASE_WEIGHT_SAVE}/${SUB_DIR}"
      CUR_MODEL_LOAD="${BASE_MODEL_LOAD}/${SUB_DIR}"
      
      mkdir -p "$CUR_RESULT" "$CUR_WEIGHT" "$CUR_MODEL_LOAD"

      # --- コマンド引数の構築 ---
      TOKENS_CMD=""
      if [ "$USE_TOKENS" = true ]; then
          TOKENS_CMD="--use_tokens"
          if [ "$NAT_SEP" = true ]; then
              TOKENS_CMD="$TOKENS_CMD --use_natural_separators"
          fi
      fi
      
      if [ "$FLOZEN_PRE" = true ]; then
          FLOZEN_CMD="--flozen_flag --peft_type None"
      else
          FLOZEN_CMD="--peft_type lora"
      fi

      EMB_FLOZEN_CMD=""
      if [ "$EMB_FLOZEN_PRE" = true ]; then
          EMB_FLOZEN_CMD="--emb_flozen_flag"
      fi

    

      echo "-------------------------------------------------------"
      echo " Running Experiment [Index: $idx]"
      echo " Dataset    : $DATASET_NAME"
      echo " Emb Type   : $TEMP_EMB_TYPE"
      echo " Weight P   : $codebook_weight_p"
      echo " Weight H   : $codebook_weight_h"
      echo " Out Dir    : $CUR_RESULT"
      echo "-------------------------------------------------------"

      python "scripts/${SCRIPT_DIR}/train_tpp_llm.py" \
        @configs/tpp_llm_nt.config \
        --dataset_path="${DATASET_PATH}" \
        --result_save_path="${CUR_RESULT}" \
        --model_weight_path="${CUR_MODEL_LOAD}" \
        --weight_path="${CUR_WEIGHT}" \
        --seed="${seed}" \
        --temporal_emb_type="${TEMP_EMB_TYPE}" \
        --device=cuda:1 \
        --codebook_loss_weight_p="${codebook_weight_p}" \
        --codebook_loss_weight_h="${codebook_weight_h}" \
        $TOKENS_CMD $FLOZEN_CMD $EMB_FLOZEN_CMD 
        
        
    done
        
  done
done


#(2)TPP-LLM+Linear+NL
# 乱数シードの設定
SEEDS=(42)
llm='llama'

WEIGHT_P_LIST=(
  0
)

WEIGHT_H_LIST=(
  0
)

# ★ 実験セットの定義（1要素）
DATASETS=( "nyc_taxi" ) 
SAVE_NAMES=("20261123/nyc_taxi" ) 
SCRIPT_DIRS=("taxi" )

# --- 共通設定 ---
TEMP_EMB_TYPES=('linear'  )
FLOSEN_FLAG_SET=( false )
USE_TOKENS_SET=( true )
NATURAL_SEP_SET=( true )
EMB_FLOZEN_SET=( false )

# =========================================================
#   Execution Loop
# =========================================================

for seed in "${SEEDS[@]}"; do

  echo "========================================="
  echo " STARTING EXPERIMENTS FOR SEED: $seed "
  echo "========================================="

  for i in "${!TEMP_EMB_TYPES[@]}"; do
    
    DATASET_NAME="${DATASETS[$i]}"
    SAVE_NAME="${SAVE_NAMES[$i]}"
    SCRIPT_DIR="${SCRIPT_DIRS[$i]}"
    TEMP_EMB_TYPE="${TEMP_EMB_TYPES[$i]}"
    
    # フラグの取り出し
    USE_TOKENS="${USE_TOKENS_SET[$i]}"
    NAT_SEP="${NATURAL_SEP_SET[$i]}"
    FLOZEN_PRE="${FLOSEN_FLAG_SET[$i]}"
    EMB_FLOZEN_PRE="${EMB_FLOZEN_SET[$i]}"
    
    # --- パス設定（共通部分） ---
    DATASET_PATH="data/${DATASET_NAME}"
    BASE_RESULT="result/${SAVE_NAME}"
    BASE_WEIGHT_SAVE="save_weight/${SAVE_NAME}"
    BASE_MODEL_LOAD="save_model/${SAVE_NAME}"

    for idx in "${!WEIGHT_P_LIST[@]}"; do
      codebook_weight_p="${WEIGHT_P_LIST[$idx]}"
      codebook_weight_h="${WEIGHT_H_LIST[$idx]}"
      
      # フォルダ名が被らないよう、重みの値もディレクトリ名に追加
      FLAG_STR="${llm}/tokens_${USE_TOKENS}_nat_${NAT_SEP}_flosen_${FLOZEN_PRE}_embflozen_${EMB_FLOZEN_PRE}_wp_${codebook_weight_p}_wh_${codebook_weight_h}"

      SUB_DIR="${TEMP_EMB_TYPE}/${FLAG_STR}/seed_${seed}"

      CUR_RESULT="${BASE_RESULT}/${SUB_DIR}"
      CUR_WEIGHT="${BASE_WEIGHT_SAVE}/${SUB_DIR}"
      CUR_MODEL_LOAD="${BASE_MODEL_LOAD}/${SUB_DIR}"
      
      mkdir -p "$CUR_RESULT" "$CUR_WEIGHT" "$CUR_MODEL_LOAD"

      # --- コマンド引数の構築 ---
      TOKENS_CMD=""
      if [ "$USE_TOKENS" = true ]; then
          TOKENS_CMD="--use_tokens"
          if [ "$NAT_SEP" = true ]; then
              TOKENS_CMD="$TOKENS_CMD --use_natural_separators"
          fi
      fi
      
      if [ "$FLOZEN_PRE" = true ]; then
          FLOZEN_CMD="--flozen_flag --peft_type None"
      else
          FLOZEN_CMD="--peft_type lora"
      fi

      EMB_FLOZEN_CMD=""
      if [ "$EMB_FLOZEN_PRE" = true ]; then
          EMB_FLOZEN_CMD="--emb_flozen_flag"
      fi

    

      echo "-------------------------------------------------------"
      echo " Running Experiment [Index: $idx]"
      echo " Dataset    : $DATASET_NAME"
      echo " Emb Type   : $TEMP_EMB_TYPE"
      echo " Weight P   : $codebook_weight_p"
      echo " Weight H   : $codebook_weight_h"
      echo " Out Dir    : $CUR_RESULT"
      echo "-------------------------------------------------------"

      python "scripts/${SCRIPT_DIR}/train_tpp_llm.py" \
        @configs/tpp_llm_nt.config \
        --dataset_path="${DATASET_PATH}" \
        --result_save_path="${CUR_RESULT}" \
        --model_weight_path="${CUR_MODEL_LOAD}" \
        --weight_path="${CUR_WEIGHT}" \
        --seed="${seed}" \
        --temporal_emb_type="${TEMP_EMB_TYPE}" \
        --device=cuda:1 \
        --codebook_loss_weight_p="${codebook_weight_p}" \
        --codebook_loss_weight_h="${codebook_weight_h}" \
        $TOKENS_CMD $FLOZEN_CMD $EMB_FLOZEN_CMD 
        
        
    done
        
  done
done


#(2)Language-TPP
# 乱数シードの設定
SEEDS=(42)
llm='llama'

WEIGHT_P_LIST=(
  0
)

WEIGHT_H_LIST=(
  0
)

# ★ 実験セットの定義（1要素）
DATASETS=( "nyc_taxi" ) 
SAVE_NAMES=("20261123/nyc_taxi" ) 
SCRIPT_DIRS=("taxi" )

# --- 共通設定 ---
TEMP_EMB_TYPES=('float'  )
FLOSEN_FLAG_SET=( false )
USE_TOKENS_SET=( true )
NATURAL_SEP_SET=( false )
EMB_FLOZEN_SET=( false )

# =========================================================
#   Execution Loop
# =========================================================

for seed in "${SEEDS[@]}"; do

  echo "========================================="
  echo " STARTING EXPERIMENTS FOR SEED: $seed "
  echo "========================================="

  for i in "${!TEMP_EMB_TYPES[@]}"; do
    
    DATASET_NAME="${DATASETS[$i]}"
    SAVE_NAME="${SAVE_NAMES[$i]}"
    SCRIPT_DIR="${SCRIPT_DIRS[$i]}"
    TEMP_EMB_TYPE="${TEMP_EMB_TYPES[$i]}"
    
    # フラグの取り出し
    USE_TOKENS="${USE_TOKENS_SET[$i]}"
    NAT_SEP="${NATURAL_SEP_SET[$i]}"
    FLOZEN_PRE="${FLOSEN_FLAG_SET[$i]}"
    EMB_FLOZEN_PRE="${EMB_FLOZEN_SET[$i]}"
    
    # --- パス設定（共通部分） ---
    DATASET_PATH="data/${DATASET_NAME}"
    BASE_RESULT="result/${SAVE_NAME}"
    BASE_WEIGHT_SAVE="save_weight/${SAVE_NAME}"
    BASE_MODEL_LOAD="save_model/${SAVE_NAME}"

    for idx in "${!WEIGHT_P_LIST[@]}"; do
      codebook_weight_p="${WEIGHT_P_LIST[$idx]}"
      codebook_weight_h="${WEIGHT_H_LIST[$idx]}"
      
      # フォルダ名が被らないよう、重みの値もディレクトリ名に追加
      FLAG_STR="${llm}/tokens_${USE_TOKENS}_nat_${NAT_SEP}_flosen_${FLOZEN_PRE}_embflozen_${EMB_FLOZEN_PRE}_wp_${codebook_weight_p}_wh_${codebook_weight_h}"

      SUB_DIR="${TEMP_EMB_TYPE}/${FLAG_STR}/seed_${seed}"

      CUR_RESULT="${BASE_RESULT}/${SUB_DIR}"
      CUR_WEIGHT="${BASE_WEIGHT_SAVE}/${SUB_DIR}"
      CUR_MODEL_LOAD="${BASE_MODEL_LOAD}/${SUB_DIR}"
      
      mkdir -p "$CUR_RESULT" "$CUR_WEIGHT" "$CUR_MODEL_LOAD"

      # --- コマンド引数の構築 ---
      TOKENS_CMD=""
      if [ "$USE_TOKENS" = true ]; then
          TOKENS_CMD="--use_tokens"
          if [ "$NAT_SEP" = true ]; then
              TOKENS_CMD="$TOKENS_CMD --use_natural_separators"
          fi
      fi
      
      if [ "$FLOZEN_PRE" = true ]; then
          FLOZEN_CMD="--flozen_flag --peft_type None"
      else
          FLOZEN_CMD="--peft_type lora"
      fi

      EMB_FLOZEN_CMD=""
      if [ "$EMB_FLOZEN_PRE" = true ]; then
          EMB_FLOZEN_CMD="--emb_flozen_flag"
      fi

    

      echo "-------------------------------------------------------"
      echo " Running Experiment [Index: $idx]"
      echo " Dataset    : $DATASET_NAME"
      echo " Emb Type   : $TEMP_EMB_TYPE"
      echo " Weight P   : $codebook_weight_p"
      echo " Weight H   : $codebook_weight_h"
      echo " Out Dir    : $CUR_RESULT"
      echo "-------------------------------------------------------"

      python "scripts/${SCRIPT_DIR}/train_tpp_llm.py" \
        @configs/tpp_llm_nt.config \
        --dataset_path="${DATASET_PATH}" \
        --result_save_path="${CUR_RESULT}" \
        --model_weight_path="${CUR_MODEL_LOAD}" \
        --weight_path="${CUR_WEIGHT}" \
        --seed="${seed}" \
        --temporal_emb_type="${TEMP_EMB_TYPE}" \
        --device=cuda:1 \
        --codebook_loss_weight_p="${codebook_weight_p}" \
        --codebook_loss_weight_h="${codebook_weight_h}" \
        $TOKENS_CMD $FLOZEN_CMD $EMB_FLOZEN_CMD 
        
        
    done
        
  done
done
