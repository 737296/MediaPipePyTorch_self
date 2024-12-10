import torch
from blazepose_landmark import BlazePoseLandmark

model = BlazePoseLandmark()
model.load_state_dict(torch.load("../blazepose_landmark.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    "blazepose_landmark.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['class', "keypoints", "seg"],
    dynamic_axes={'input': {0: 'batch_size'}, 'class': {0: 'batch_size'}, 'keypoints': {0: 'batch_size'},
                  'seg': {0: 'batch_size'}}
)

import onnx

onnx_model = onnx.load("blazepose_landmark.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX 模型已成功导出！")
