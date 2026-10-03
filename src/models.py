"""Models (Keras 3): standard U-Net, EfficientNet-B0 U-Net, Attention U-Net. All fully convolutional
(input (None,None,3), side lengths multiple of 32) with 1-channel sigmoid output."""
import keras
from keras import layers


def _conv_block(x, f, dropout=0.0):
    for _ in range(2):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    if dropout:
        x = layers.SpatialDropout2D(dropout)(x)
    return x


def build_unet(filters=(16, 32, 64, 128), bottleneck=256, dropout=0.1, attention=False):
    inp = layers.Input((None, None, 3))
    x, skips = inp, []
    for f in filters:
        x = _conv_block(x, f)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
    x = _conv_block(x, bottleneck, dropout)
    for f, s in zip(reversed(filters), reversed(skips)):
        x = layers.UpSampling2D(interpolation="bilinear")(x)
        x = layers.Conv2D(f, 2, padding="same")(x)
        if attention:
            s = _attention_gate(s, x, f // 2)
        x = layers.Concatenate()([x, s])
        x = _conv_block(x, f)
    out = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="attention_unet" if attention else "unet")


def _attention_gate(skip, gate, inter):
    a = layers.Conv2D(inter, 1)(skip)
    b = layers.Conv2D(inter, 1)(gate)
    psi = layers.Activation("relu")(layers.Add()([a, b]))
    psi = layers.Conv2D(1, 1, activation="sigmoid")(psi)
    return layers.Multiply()([skip, psi])


def build_effnet_unet(decoder_filters=(128, 64, 32, 16, 8), pretrained=False, dropout=0.1):
    """U-Net with EfficientNet-B0 encoder. `pretrained=True` needs ImageNet weights to be downloadable."""
    inp = layers.Input((None, None, 3))
    enc = keras.applications.EfficientNetB0(include_top=False, weights="imagenet" if pretrained else None,
                                            input_tensor=layers.Rescaling(255.0)(inp))
    skip_names = ["block6a_expand_activation",  # 1/16
                  "block4a_expand_activation",  # 1/8
                  "block3a_expand_activation",  # 1/4
                  "block2a_expand_activation"]  # 1/2
    skips = [enc.get_layer(n).output for n in skip_names]
    x = enc.get_layer("top_activation").output  # 1/32
    for i, f in enumerate(decoder_filters):
        x = layers.UpSampling2D(interpolation="bilinear")(x)
        if i < len(skips):
            x = layers.Concatenate()([x, skips[i]])
        x = _conv_block(x, f, dropout if i == 0 else 0.0)
    out = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="effnetb0_unet")


def build_model(cfg):
    m = cfg["model"]
    if m == "unet":
        return build_unet(tuple(cfg.get("filters", (16, 32, 64, 128))), cfg.get("bottleneck", 256), cfg.get("dropout", 0.1))
    if m == "attention_unet":
        return build_unet(tuple(cfg.get("filters", (16, 32, 64, 128))), cfg.get("bottleneck", 256), cfg.get("dropout", 0.1), attention=True)
    if m == "effnet_unet":
        return build_effnet_unet(pretrained=cfg.get("pretrained", False), dropout=cfg.get("dropout", 0.1))
    raise ValueError(m)
