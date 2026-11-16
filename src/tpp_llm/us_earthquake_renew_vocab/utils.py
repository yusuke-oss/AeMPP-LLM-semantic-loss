def get_prompt(
    dataset_name: str, 
    event_time_first: bool,
    temporal_emb_type: str = 'MLP' , # Added: Branching by embedding type
    use_natural_separators: bool = False  # ← これを追加！
) -> str:
    """
    Get the prompt for the TPP-LLM
    """
    sequence_descriptions = {
        "us_earthquake": "You are given a sequence of earthquake events recorded in the United States.",
    }

    # Modified: Dynamically change prompt descriptions for TPE and MLP
    if temporal_emb_type == 'positional':
        # For TPE (exclude descriptions of special tokens)
        if event_time_first:
            event_description = "Each event in the sequence lists the timestamp followed by the magnitude classification (large or medium or small), the magnitude value, the depth value, the latitude value, and the longitude value."
        else:
            event_description = "Each event in the sequence lists the timestamp followed by the magnitude classification (large or medium or small), the magnitude value, the depth value, the latitude value, and the longitude value."
    elif temporal_emb_type == 'MLP_fusion_k':
        if event_time_first:
            event_description = "Each event in the sequence lists the integrated value of timestamp, magnitude, depth, latitude, and longitude by a trainable MLP followed by the magnitude classification (large or medium or small)."
        else:
            event_description = "Each event in the sequence lists the integrated value of timestamp, magnitude, depth, latitude, and longitude by a trainable MLP followed by the magnitude classification (large or medium or small)."

    elif temporal_emb_type in ['MLP_fusion']:
        if event_time_first:
            event_description = "Each event in the sequence lists the integrated value of timestamp, magnitude, depth, latitude, and longitude by a trainable MLP."
        else:
            event_description = "Each event in the sequence lists the integrated value of timestamp, magnitude, and depth by a trainable MLP. "
    elif temporal_emb_type in ['MLP_token_renew_vocab_topk_loss_ph','MLP_token_renew_vocab_loss_ph_scaled','MLP_fusion_renew','MLP_token_renew_loss','MLP_token_renew_loss_p','MLP_token_renew_loss_h','MLP_token_renew_loss_ph','MLP_token_renew_topk_loss_p', 'MLP_token_renew_topk_loss_h', 'MLP_token_renew_topk_loss_ph','MLP_token_renew_vocab_loss_ph']:
        if event_time_first:
            event_description = "Each event is represented by a single event token obtained by compressing its timestamp, magnitude, depth, latitude, longitude, and event-type information through an MLP."
        else:
            event_description = "Each event is represented by a single event token obtained by compressing its timestamp, magnitude, depth, latitude, longitude, and event-type information through an MLP."
    elif temporal_emb_type in ['MLP_token_renew_loss_ph_label']:
        if event_time_first:
            event_description = "Each event in the sequence is represented by the integrated value of timestamp, magnitude, depth, latitude, and longitude through a trainable MLP, followed by the event-type information (large, medium, or small)."
        else:
            event_description = "Each event in the sequence is represented by the integrated value of timestamp, magnitude, depth, latitude, and longitude through a trainable MLP, followed by the event-type information (large, medium, or small)."
        
    #Each event in the sequence lists the integrated value of timestamp, magnitude, depth, and magnitude classification (large or medium or small) by a trainable MLP.
    elif temporal_emb_type in ['MLP_fusion_semantic']:
        #if event_time_first:
        #    event_description = "Each event in the sequence lists the integrated value of timestamp, magnitude, and depth by a trainable MLP, infused with the magnitude classification (large or medium or small)."
        #else:
        #    event_description = "Each event in the sequence lists the integrated value of timestamp, magnitude, and depth by a trainable MLP, infused with the magnitude classification (large or medium or small)."

        if event_time_first:
            event_description = "Each event in the sequence lists the integrated value of timestamp, magnitude, depth, latitude, and longitude by a trainable MLP."
        else:
            event_description = "Each event in the sequence lists the integrated value of timestamp, magnitude, depth, latitude, and longitude by a trainable MLP."
    
    elif temporal_emb_type == 'main':
        event_descriptions = {
            "us_earthquake": {
                "event_type_first": "Each event in the sequence lists the magnitude classification (large or small) followed by the timestamp.",
                "event_time_first": "Each event in the sequence lists the timestamp followed by the magnitude classification (large or small)."
                },
            }
        event_description = event_descriptions.get(dataset_name, {}).get(
            "event_time_first" if event_time_first else "event_type_first", "")

    else:
        # For MLP (include descriptions of special tokens as before)
        if use_natural_separators==True:
            event_descriptions = {
            "us_earthquake": {
                "event_type_first": "Each event in the sequence lists the magnitude classification (large or medium or small) followed by the magnitude, followed by the depth, followed by the timestamp.",
                "event_time_first": "Each event in the sequence lists the timestamp followed by the magnitude classification (large or medium or small), the magnitude value, the depth value, the latitude value, and the longitude value."
                },
            }
        else:
            event_descriptions = {
            "us_earthquake": {
                "event_type_first": "Each event in the sequence lists the magnitude classification (large or medium or small) followed by the magnitude, followed by the depth, followed by the timestamp. <|time_prefix|> is Prefix token for event timestamp, <|type_prefix|> is Prefix token for event type, <|magnitude_prefix|> is Prefix token for event magnitude, <|depth_prefix|> is Prefix token for event depth.",
                "event_time_first": "Each event in the sequence lists the timestamp followed by the magnitude classification (large or medium or small), the magnitude value, the depth value, the latitude value, and the longitude value. <|time_prefix|> is Prefix token for event timestamp, <|type_prefix|> is Prefix token for event type, <|magnitude_prefix|> is Prefix token for event magnitude, <|depth_prefix|> is Prefix token for event depth, <|latitude_prefix|> is Prefix token for event latitude , <|longitude_prefix|> is Prefix token for event longitude."
                },
            }
            '''
        else:
            event_descriptions = {
            "us_earthquake": {
                "event_type_first": "Each event in the sequence lists the magnitude classification (large or medium or small) followed by the magnitude, followed by the depth, followed by the timestamp. <|time_prefix|> is Prefix token for event timestamp, <|type_prefix|> is Prefix token for event type, <|magnitude_prefix|> is Prefix token for event magnitude, <|depth_prefix|> is Prefix token for event depth.",
                "event_time_first": "Each event in the sequence lists the timestamp followed by the magnitude classification (large or medium or small), followed by the magnitude, followed by the depth. <|time_prefix|> is Prefix token for event timestamp, <|type_prefix|> is Prefix token for event type, <|magnitude_prefix|> is Prefix token for event magnitude, <|depth_prefix|> is Prefix token for event depth."
                },
            }
'''
        event_description = event_descriptions.get(dataset_name, {}).get(
            "event_time_first" if event_time_first else "event_type_first", "")
    if temporal_emb_type in ['MLP_token_renew_vocab_topk_loss_ph','MLP_token_renew_vocab_loss_ph_scaled','MLP_fusion_renew','MLP_token_renew_loss','MLP_token_renew_loss_p','MLP_token_renew_loss_h','MLP_token_renew_loss_ph','MLP_token_renew_topk_loss_p', 'MLP_token_renew_topk_loss_h', 'MLP_token_renew_topk_loss_ph','MLP_token_renew_vocab_loss_ph']:
            task_descriptions = {
            "event_type_first": "Based on the observed sequence, predict the time and event type of the next earthquake event.",
            "event_time_first": "Based on the observed sequence, predict the time and event type of the next earthquake event."
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