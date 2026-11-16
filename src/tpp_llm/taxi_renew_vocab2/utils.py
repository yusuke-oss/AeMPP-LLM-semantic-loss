def get_prompt(
    dataset_name: str, 
    event_time_first: bool,
    temporal_emb_type: str = 'MLP', 
    use_natural_separators: bool = False
) -> str:
    """
    Get the prompt for the TPP-LLM
    """
    sequence_descriptions = {
        "us_earthquake": "You are given a sequence of earthquake events recorded in the United States.",
        "nyc_taxi": "You are given a sequence of taxi trips taken in New York City.",
    }

    # ====================================================================
    # 1. Positional Encoding の場合（特殊トークンの説明を省いたクリーンな文章）
    # ====================================================================
    if temporal_emb_type == 'positional':
        if dataset_name == 'us_earthquake':
            if event_time_first:
                event_description = "Each event in the sequence lists the timestamp followed by the magnitude classification (large or medium or small), the magnitude value, and the depth value."
            else:
                event_description = "Each event in the sequence lists the magnitude classification (large or medium or small) followed by the magnitude value, the depth value, and the timestamp."
        
        elif dataset_name == 'nyc_taxi':
            if event_time_first:
                event_description = "Each event in the sequence lists the timestamp followed by the pick-up or drop-off location, the latitude value, the longitude value, and the distance value."
            else:
                event_description = "Each event in the sequence lists the pick-up or drop-off location followed by the latitude value, the longitude value, the distance value, and the timestamp."

    # ====================================================================
    # 2. MLP_fusion の場合（数値を統合し、タクシーのみTypeテキストを残す）
    # ====================================================================
    elif temporal_emb_type == 'MLP_fusion_k':
        if dataset_name == 'us_earthquake':
            event_description = "Each event in the sequence lists the integrated value of timestamp, magnitude, and depth by a trainable MLP."
        elif dataset_name == 'nyc_taxi':
            # ★ MLP_fusion時はテキスト説明を完全に排除し、数値の融合のみで推論させる
            event_description = "Each event in the sequence lists the integrated value of timestamp, latitude, longitude, and distance by a trainable MLP followed by the pick-up or drop-off location.."
    

    elif temporal_emb_type == 'MLP_fusion':
        if dataset_name == 'us_earthquake':
            event_description = "Each event in the sequence lists the integrated value of timestamp, magnitude, and depth by a trainable MLP."
        elif dataset_name == 'nyc_taxi':
            # ★ MLP_fusion時はテキスト説明を完全に排除し、数値の融合のみで推論させる
            event_description = "Each event in the sequence lists the integrated value of timestamp, latitude, longitude, and distance by a trainable MLP."
    elif temporal_emb_type in['MLP_token_renew_vocab_topk_loss_ph','MLP_token_renew_vocab_loss_ph_scaled','MLP_token_renew_topk_loss_ph', 'MLP_fusion_renew','MLP_token_renew_loss','MLP_token_renew_loss_p','MLP_token_renew_loss_h','MLP_token_renew_loss_ph','MLP_token_renew_topk_loss_p', 'MLP_token_renew_topk_loss_h', 'MLP_token_renew_topk_loss_ph','MLP_token_renew_vocab_loss_ph']:
        if dataset_name == 'us_earthquake':
            event_description = "Each event in the sequence lists the integrated value of timestamp, magnitude, and depth by a trainable MLP."
        elif dataset_name == 'nyc_taxi':
            # ★ MLP_fusion時はテキスト説明を完全に排除し、数値の融合のみで推論させる
            event_description = "Each event is represented by a single event token obtained by compressing its timestamp, latitude, longitude, distance, and event-type information through an MLP."
    #Each event in the sequence lists the integrated value of timestamp, latitude, longitude, distance, and event type by a trainable MLP.
    elif temporal_emb_type == 'main':
        event_descriptions = {
            "nyc_taxi": {
                "event_type_first": "Each event in the sequence lists the pick-up or drop-off location followed by the timestamp.",
                "event_time_first": "Each event in the sequence lists the timestamp followed by the pick-up or drop-off location."
                }
            }
        # 辞書から該当する説明文を取得
        event_description = event_descriptions.get(dataset_name, {}).get(
            "event_time_first" if event_time_first else "event_type_first", "")

    # ====================================================================
    # 3. 通常の MLP / float embeddings の場合（自然言語区切り or トークン区切り）
    # ====================================================================
    # 3. 通常の MLP / float embeddings の場合（自然言語区切り or トークン区切り）
    # ====================================================================
    else:
        if use_natural_separators == True:
            event_descriptions = {
                "us_earthquake": {
                    "event_type_first": "Each event in the sequence lists the magnitude classification (large or medium or small) followed by the magnitude value, the depth value, and the timestamp.",
                    "event_time_first": "Each event in the sequence lists the timestamp followed by the magnitude classification (large or medium or small), the magnitude value, and the depth value."
                },
                "nyc_taxi": {
                    "event_type_first": "Each event in the sequence lists the pick-up or drop-off location followed by the latitude value, the longitude value, the distance value, and the timestamp.",
                    "event_time_first": "Each event in the sequence lists the timestamp followed by the pick-up or drop-off location, the latitude value, the longitude value, and the distance value."
                }
            }
        else:
            # 特殊トークン（<|..._prefix|>）を使用する場合
            event_descriptions = {
                "us_earthquake": {
                    "event_type_first": "Each event in the sequence lists the magnitude classification (large or medium or small) followed by the magnitude value, the depth value, and the timestamp. <|time_prefix|> is Prefix token for event timestamp, <|type_prefix|> is Prefix token for event type, <|magnitude_prefix|> is Prefix token for event magnitude, <|depth_prefix|> is Prefix token for event depth.",
                    "event_time_first": "Each event in the sequence lists the timestamp followed by the magnitude classification (large or medium or small), the magnitude value, and the depth value. <|time_prefix|> is Prefix token for event timestamp, <|type_prefix|> is Prefix token for event type, <|magnitude_prefix|> is Prefix token for event magnitude, <|depth_prefix|> is Prefix token for event depth."
                },
                "nyc_taxi": {
                    "event_type_first": "Each event in the sequence lists the pick-up or drop-off location followed by the latitude value, the longitude value, the distance value, and the timestamp. <|time_prefix|> is Prefix token for event timestamp, <|type_prefix|> is Prefix token for event type, <|latitude_prefix|> is Prefix token for event latitude, <|longitude_prefix|> is Prefix token for event longitude, <|distance_prefix|> is Prefix token for event distance.",
                    "event_time_first": "Each event in the sequence lists the timestamp followed by the pick-up or drop-off location, the latitude value, the longitude value, and the distance value. <|time_prefix|> is Prefix token for event timestamp, <|type_prefix|> is Prefix token for event type, <|latitude_prefix|> is Prefix token for event latitude, <|longitude_prefix|> is Prefix token for event longitude, <|distance_prefix|> is Prefix token for event distance."
                }
            }
        # 辞書から該当する説明文を取得
        event_description = event_descriptions.get(dataset_name, {}).get(
            "event_time_first" if event_time_first else "event_type_first", "")
        
    if temporal_emb_type in ['MLP_token_renew_vocab_topk_loss_ph','MLP_token_renew_vocab_loss_ph_scaled','MLP_token_renew_topk_loss_ph','MLP_fusion_renew','MLP_token_renew_loss','MLP_token_renew_loss_p','MLP_token_renew_loss_h','MLP_token_renew_loss_ph','MLP_token_renew_topk_loss_p', 'MLP_token_renew_topk_loss_h', 'MLP_token_renew_topk_loss_ph','MLP_token_renew_vocab_loss_ph']:
            task_descriptions = {
        "event_type_first": "Based on the observed sequence, predict the time and event type of the next taxi trip event.",
        "event_time_first": "Based on the observed sequence, predict the time and event type of the next taxi trip event."
    }
    else:
        task_descriptions = {
            "event_type_first": "Based on this sequence, predict the next event type and the corresponding time.",
            "event_time_first": "Based on this sequence, predict the next event time and the corresponding type."
        }
    
    
    
    if dataset_name not in sequence_descriptions:
        return "Dataset not recognized."

    sequence_description = sequence_descriptions[dataset_name]
    task_description = task_descriptions["event_time_first" if event_time_first else "event_type_first"]

    return f"{sequence_description} {event_description} {task_description} "