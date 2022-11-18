import torch
import sys
sys.path.append(r'D:\Workplaces\Python\BKC\VideoFaceRecognition')
import facenet_pytorch
import numpy as np

def get_dummy_inputs(batch_size=32, n_channels=3, image_size=160):
    return torch.rand(batch_size, n_channels, image_size, image_size)


model = facenet_pytorch.InceptionResnetV1(pretrained='vggface2').eval()

dummy_inputs = get_dummy_inputs(batch_size=1)
input_names = ['input']
output_names = ['output']

save_path = 'saved/face_recognition_facenet.onnx'
torch.onnx.export(
    model,
    dummy_inputs,
    save_path,
    verbose=False,
    input_names=input_names,
    output_names=output_names,
    export_params=True
)

# Test onnx model
import onnxruntime as onnxrt
onnx_session = onnxrt.InferenceSession(save_path)
onnx_inputs = {
    onnx_session.get_inputs()[0].name: dummy_inputs.numpy()
}
onnx_output = onnx_session.run(None, onnx_inputs)[0]
actual_output = model(torch.tensor(dummy_inputs)).detach().numpy()
print('Actual Output', actual_output)
print('ONNX Output: ', onnx_output)
print('Compare output')

err = np.abs(onnx_output - actual_output)
e_max = np.max(err)
e_min = np.min(err)

print('- Min: ', e_min)
print('- Max: ', e_max)