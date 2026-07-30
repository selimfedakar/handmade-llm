# What goes in here

Nothing that is committed. `python 08_ship_ios/bundle_model.py` fills this
directory with the tokenizer and the model, and `.gitignore` keeps all of it
out of the repository:

    tokenizer.json                   what the app encodes text with
    model-quantized.safetensors      14.9 MiB, the model that ships
    model-quantized.json             its meta.json, renamed to sit beside it
    model-float32.safetensors        95 MiB, only for the paired measurement
    model-float32.json

The directory itself is tracked, and this file is the reason: Xcode's
synchronized groups want the folder to exist, and an app target that references
a folder git has never heard of does not build on a fresh clone.
