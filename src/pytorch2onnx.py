import argparse
import torch
import torch.nn as nn


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        if act_type == 'prelu':
            self.body.append(nn.PReLU(num_parameters=num_feat))
        else:
            self.body.append(nn.ReLU(inplace=True))

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == 'prelu':
                self.body.append(nn.PReLU(num_parameters=num_feat))
            else:
                self.body.append(nn.ReLU(inplace=True))

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = torch.nn.functional.interpolate(x, scale_factor=self.upscale, mode='nearest')
        return out + base


def main(args):
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    ckpt = torch.load(args.input, map_location='cpu')
    key = 'params_ema' if 'params_ema' in ckpt else 'params'
    model.load_state_dict(ckpt[key])
    model.train(False)
    model.cpu().eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        torch.onnx.export(
            model, x, args.output,
            opset_version=11,
            export_params=True,
            dynamic_axes={
                'input': {2: 'height', 3: 'width'},
                'output': {2: 'height', 3: 'width'}
            },
            input_names=['input'],
            output_names=['output']
        )
    print("Done:", args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input .pth path')
    parser.add_argument('--output', type=str, required=True, help='Output .onnx path')
    args = parser.parse_args()
    main(args)