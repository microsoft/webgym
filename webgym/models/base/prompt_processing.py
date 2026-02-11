import base64

def batch_get_vllm_prompts(system_prompts, user_prompts, image_paths, force_base64=False):
    """
    Create batch of vLLM-compatible prompts from system prompts, user prompts, and image paths.
    Uses file:// URLs for better performance when possible, falls back to base64 encoding if needed.

    Args:
        system_prompts: List of system prompts
        user_prompts: List of user prompts
        image_paths: List of current image paths
        force_base64: If True, always use base64 encoding (for compatibility)
    Returns:
        List of message lists formatted for vLLM
    """
    result = []

    for i in range(len(system_prompts)):
        # Create the message structure for this item
        messages = []

        # Add system message
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompts[i]
                }
            ]
        })

        # Add user message with image
        user_content = [
            {
                "type": "text",
                "text": user_prompts[i]
            }
        ]

        # Add current image if provided
        if i < len(image_paths) and image_paths[i]:
            if force_base64:
                # Use base64 encoding for compatibility
                with open(image_paths[i], "rb") as image_file:
                    img_data = base64.b64encode(image_file.read()).decode('utf-8')
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_data}"}
                })
            else:
                # Use file URL format for better performance (requires vLLM config)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"file://{image_paths[i]}"}
                })

        messages.append({
            "role": "user",
            "content": user_content
        })

        result.append(messages)

    return result

def batch_get_hf_prompts(system_prompts, user_prompts, image_paths, use_base64=True):
    """
    Create batch of HuggingFace-compatible prompts from system prompts, user prompts, and image paths.

    Args:
        system_prompts: List of system prompts
        user_prompts: List of user prompts
        image_paths: List of current image paths
        use_base64: Whether to encode images as base64 (True for API calls, False for local training - much faster!)
    Returns:
        List of message lists formatted for HuggingFace
    """
    try:
        import deepspeed
        rank = deepspeed.comm.get_rank()
    except:
        rank = 0

    result = []
    total = len(system_prompts)

    for i in range(total):
        # Print progress every 1000 samples on rank 0
        if rank == 0 and i > 0 and i % 1000 == 0:
            pct = (i / total) * 100
            print(f"    Processing images: {i}/{total} ({pct:.1f}%)...")
        # Create the message structure for this item
        messages = []

        # Add system message
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompts[i]
                }
            ]
        })

        # Add user message with current image only
        user_content = [
            {
                "type": "text",
                "text": user_prompts[i]
            }
        ]

        # Add current image
        if i < len(image_paths) and image_paths[i]:
            if use_base64:
                # Slow path: encode to base64 (for API calls)
                from webgym.utils import encode_image_to_base64
                current_image_url = encode_image_to_base64(image_paths[i])
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": current_image_url}
                })
            else:
                # Fast path for training: use file path directly (no loading/encoding needed)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"file://{image_paths[i]}"}
                })

        messages.append({
            "role": "user",
            "content": user_content
        })

        result.append(messages)

    return result