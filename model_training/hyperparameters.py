from enums import Mode


class HyperParams:
    k_class = 15
    num_epochs = 50
    batch_size = 16
    k_hidden = 150
    k_layers = 1
    train_size = 100
    device = 0
    net = 'wb'
    mode = Mode.CLIPS.value
    zscore = True
