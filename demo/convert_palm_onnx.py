import torch
from blazepalm import BlazePalm

model = BlazePalm()
model.load_state_dict(torch.load("../blazepalm.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    "blazepalm.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['rec', "class"],
    dynamic_axes={'input': {0: 'batch_size'}, 'rec': {0: 'batch_size'}, 'class': {0: 'batch_size'}}
)

import onnx

onnx_model = onnx.load("blazepalm.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX 模型已成功导出！")
