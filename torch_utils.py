feats = {}


def hook_func(m, inp, op):
    if isinstance(op, tuple):
        op = op[1][0]
    feats['activations'] = op.detach()


visualisation = {}


def hook_fn(m, i, o):
    visualisation[m] = o


activation = {}


def getActivation(name):
    def hook(model, input, output):
        activation['lstm_activations'] = input[0].detach()
        activation['linear_activations'] = output.detach()
    return hook
