import torch

# define a floating point model
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        # QuantStub to convert tensors from fp32 to quantized int8
        self.quant_layer = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(5, 5, 5)
        self.relu = torch.nn.ReLU()
        # DeQuantStub to convert tensors from quantized int8 to fp32
        self.dequant_layer = torch.quantization.DeQuantStub()


    def forward(self, x):
        x = self.fc(x)
        return x

# create a model instance
model_fp32 = M()
model = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

# create a quantized model instance
model_int8 = torch.ao.quantization.convert(model_fp32)

print(model_int8)

parameters = [i.detach().cpu().numpy() for i in model_int8.parameters()]
print([i.dtype for i in parameters])
print(parameters)

quantized_model = torch.quantization.quantize_dynamic(model_fp32, {torch.nn.Linear}, dtype=torch.qint8)

parameters = [i.detach().cpu().numpy() for i in quantized_model.parameters()]
print([i.dtype for i in parameters])
print(parameters)