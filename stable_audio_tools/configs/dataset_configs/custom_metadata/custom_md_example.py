def get_custom_metadata(info, audio):

    # Use relative path as the prompt
    # return {"prompt": info["relpath"]}
    return {"control_signal": audio}
    # return {"prompt": info["relpath"], "control_signal": audio}
    # return {"prompt": info["relpath"], "control_signal": info["path"]}