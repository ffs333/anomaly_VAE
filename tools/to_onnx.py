import torch.onnx
import onnxruntime
import numpy as np

from model.ae_model import AE
from model.ae_model_light import AELight


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


folder = './checkpoints'
path_torch_model = f'{folder}/model_weights.ckpt'
name = path_torch_model.split('/')[-1].split('.')[0]
path_test_input = None  # or None for random input
path_to_save_onnx = f'{folder}/onnx/{name}.onnx'

batch_size = 1
input_size = [32, 256]
model = AELight(input_size, batch_size=batch_size)

transfer = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if transfer:
    weigths = torch.load(path_torch_model, map_location=device)
    model.load_state_dict(weigths)
model.eval()

if path_test_input:
    opa = torch.load(path_test_input)
    x = opa[4].unsqueeze(1)
else:
    x = torch.randn(batch_size, 1, input_size[0], input_size[1], requires_grad=True)

# meta = torch.randn(batch_size, 7, requires_grad=True)
torch_out = model.forward(x)

torch.onnx.export(model,                     # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  path_to_save_onnx,         # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input_1', 'input_2'],   # the model's input names
                  output_names=['output'],              # the model's output names
                  dynamic_axes={'input_1': {0: 'batch_size'},    # variable length axes
                                'input_2': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})


ort_session = onnxruntime.InferenceSession(path_to_save_onnx)

ort_outs = ort_session.run(None, {
    'input_1': to_numpy(x),
    # 'input_2': to_numpy(meta)
    })

torch_out_num = to_numpy(torch_out[0])
onxx_out_num = ort_outs[0]

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(torch_out_num, onxx_out_num, rtol=1e-01, atol=1e-01)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")