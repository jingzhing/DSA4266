import timm

def build_model(model_name):
    return timm.create_model(model_name, pretrained=True, num_classes=1)