from .iresnet_advBCT import iresnet18_advBCT, iresnet34_advBCT, iresnet50_advBCT
from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200



def get_model_advBCT(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18_advBCT(False, **kwargs)
    elif name == "r34":
        return iresnet34_advBCT(False, **kwargs)
    elif name == "r50":
        return iresnet50_advBCT(False, **kwargs)
    elif name == "r100":
        return iresnet100_advBCT(False, **kwargs)
    elif name == "r200":
        return iresnet200_advBCT(False, **kwargs)

def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False, **kwargs)
