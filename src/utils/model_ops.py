import torch
import torch.nn.functional as F


def upsample_disp(disp, mask, factor=4, disp_up=False):
    """Convex upsampling used by the VisDepth recurrent refinement."""
    batch, channels, height, width = disp.shape
    mask = mask.view(batch, 1, 9, factor, factor, height, width)
    mask = torch.softmax(mask, dim=2)
    patches = F.unfold(factor * disp if disp_up else disp, [3, 3], padding=1)
    patches = patches.view(batch, channels, 9, 1, 1, height, width)
    output = torch.sum(mask * patches, dim=2)
    output = output.permute(0, 1, 4, 2, 5, 3)
    return output.reshape(batch, channels, factor * height, factor * width)


def freeze_model(model):
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
