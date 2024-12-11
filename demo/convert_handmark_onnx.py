import torch
from blazehand_landmark import BlazeHandLandmark

model = BlazeHandLandmark()
model.load_state_dict(torch.load("../blazehand_landmark.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    "blazehand_landmark.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['hand_flag', "handed", "landmarks"],
    dynamic_axes={'input': {0: 'batch_size'}, 'hand_flag': {0: 'batch_size'}, 'handed': {0: 'batch_size'},
                  'landmarks': {0: 'batch_size'}}
)

import onnx

onnx_model = onnx.load("blazehand_landmark.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX 模型已成功导出！")
