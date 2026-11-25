def get_prompt(
    dataset_name: str, 
    temporal_emb_type: str = 'MLP', 
    use_natural_separators: bool = False
) -> str:
    """
    Get the prompt for the TPP-LLM
    """
    sequence_descriptions = {
        "nyc_taxi": "You are given a sequence of taxi trips taken in New York City.",
    }

    # ====================================================================
    # 1. Positional Encoding の場合（特殊トークンの説明を省いたクリーンな文章）
    # ====================================================================
    
    if temporal_emb_type in['MLP_token_renew_vocab_topk_loss_ph','MLP_token_renew_vocab_loss_ph_scaled','MLP_token_renew_topk_loss_ph', 'MLP_fusion_renew','MLP_token_renew_loss','MLP_token_renew_loss_p','MLP_token_renew_loss_h','MLP_token_renew_loss_ph','MLP_token_renew_topk_loss_p', 'MLP_token_renew_topk_loss_h', 'MLP_token_renew_topk_loss_ph','MLP_token_renew_vocab_loss_ph']:
        event_description = "Each event is represented by a single event token obtained by compressing its timestamp, latitude, longitude, distance, and event-type information through an MLP."
    #Each event in the sequence lists the integrated value of timestamp, latitude, longitude, distance, and event type by a trainable MLP.

    elif temporal_emb_type == 'main':
        event_description = "Each event in the sequence lists the timestamp followed by the pick-up or drop-off location."

    else:
        if use_natural_separators == True:
            event_description = "Each event in the sequence lists the timestamp followed by the pick-up or drop-off location, the latitude value, the longitude value, and the distance value."
        else:
            # 特殊トークン（<|..._prefix|>）を使用する場合
            event_description = "Each event in the sequence lists the timestamp followed by the pick-up or drop-off location, the latitude value, the longitude value, and the distance value. <|time_prefix|> is Prefix token for event timestamp, <|type_prefix|> is Prefix token for event type, <|latitude_prefix|> is Prefix token for event latitude, <|longitude_prefix|> is Prefix token for event longitude, <|distance_prefix|> is Prefix token for event distance."
        
    if temporal_emb_type in ['MLP_token_renew_vocab_topk_loss_ph','MLP_token_renew_vocab_loss_ph_scaled','MLP_token_renew_topk_loss_ph','MLP_fusion_renew','MLP_token_renew_loss','MLP_token_renew_loss_p','MLP_token_renew_loss_h','MLP_token_renew_loss_ph','MLP_token_renew_topk_loss_p', 'MLP_token_renew_topk_loss_h', 'MLP_token_renew_topk_loss_ph','MLP_token_renew_vocab_loss_ph']:
        task_description = "Based on the observed sequence, predict the time and event type of the next taxi trip event."
    else:
        task_description = "Based on this sequence, predict the next event time and the corresponding type."
        
    if dataset_name not in sequence_descriptions:
        return "Dataset not recognized."

    sequence_description = sequence_descriptions[dataset_name]

    return f"{sequence_description} {event_description} {task_description} "