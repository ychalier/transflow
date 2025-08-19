import math
import re
from typing import cast

import cupy
import numpy
import torch

from ...types import Rgb, Flow


torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True


backwarp_tenGrid = {}
netNetwork = None


kernel_Correlation_rearrange = '''
    extern "C" __global__ void kernel_Correlation_rearrange(
        const int n,
        const float* input,
        float* output
    ) {
      int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

      if (intIndex >= n) {
        return;
      }

      int intSample = blockIdx.z;
      int intChannel = blockIdx.y;

      float fltValue = input[(((intSample * SIZE_1(input)) + intChannel) * SIZE_2(input) * SIZE_3(input)) + intIndex];

      __syncthreads();

      int intPaddedY = (intIndex / SIZE_3(input)) + 3*{{intStride}};
      int intPaddedX = (intIndex % SIZE_3(input)) + 3*{{intStride}};
      int intRearrange = ((SIZE_3(input) + 6*{{intStride}}) * intPaddedY) + intPaddedX;

      output[(((intSample * SIZE_1(output) * SIZE_2(output)) + intRearrange) * SIZE_1(input)) + intChannel] = fltValue;
    }
'''


kernel_Correlation_updateOutput = '''
    extern "C" __global__ void kernel_Correlation_updateOutput(
      const int n,
      const float* rbot0,
      const float* rbot1,
      float* top
    ) {
      extern __shared__ char patch_data_char[];
      
      float *patch_data = (float *)patch_data_char;
      
      // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
      int x1 = (blockIdx.x + 3) * {{intStride}};
      int y1 = (blockIdx.y + 3) * {{intStride}};
      int item = blockIdx.z;
      int ch_off = threadIdx.x;
      
      // Load 3D patch into shared shared memory
      for (int j = 0; j < 1; j++) { // HEIGHT
        for (int i = 0; i < 1; i++) { // WIDTH
          int ji_off = (j + i) * SIZE_3(rbot0);
          for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
            int idx1 = ((item * SIZE_1(rbot0) + y1+j) * SIZE_2(rbot0) + x1+i) * SIZE_3(rbot0) + ch;
            int idxPatchData = ji_off + ch;
            patch_data[idxPatchData] = rbot0[idx1];
          }
        }
      }
      
      __syncthreads();
      
      __shared__ float sum[32];
      
      // Compute correlation
      for (int top_channel = 0; top_channel < SIZE_1(top); top_channel++) {
        sum[ch_off] = 0;
      
        int s2o = (top_channel % 7 - 3) * {{intStride}};
        int s2p = (top_channel / 7 - 3) * {{intStride}};
        
        for (int j = 0; j < 1; j++) { // HEIGHT
          for (int i = 0; i < 1; i++) { // WIDTH
            int ji_off = (j + i) * SIZE_3(rbot0);
            for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
              int x2 = x1 + s2o;
              int y2 = y1 + s2p;
              
              int idxPatchData = ji_off + ch;
              int idx2 = ((item * SIZE_1(rbot0) + y2+j) * SIZE_2(rbot0) + x2+i) * SIZE_3(rbot0) + ch;
              
              sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
            }
          }
        }
        
        __syncthreads();
        
        if (ch_off == 0) {
          float total_sum = 0;
          for (int idx = 0; idx < 32; idx++) {
            total_sum += sum[idx];
          }
          const int sumelems = SIZE_3(rbot0);
          const int index = ((top_channel*SIZE_2(top) + blockIdx.y)*SIZE_3(top))+blockIdx.x;
          top[index + item*SIZE_1(top)*SIZE_2(top)*SIZE_3(top)] = total_sum / (float)sumelems;
        }
      }
    }
'''


kernel_Correlation_updateGradOne = '''
    #define ROUND_OFF 50000

    extern "C" __global__ void kernel_Correlation_updateGradOne(
      const int n,
      const int intSample,
      const float* rbot0,
      const float* rbot1,
      const float* gradOutput,
      float* gradOne,
      float* gradTwo
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
      int n = intIndex % SIZE_1(gradOne); // channels
      int l = (intIndex / SIZE_1(gradOne)) % SIZE_3(gradOne) + 3*{{intStride}}; // w-pos
      int m = (intIndex / SIZE_1(gradOne) / SIZE_3(gradOne)) % SIZE_2(gradOne) + 3*{{intStride}}; // h-pos
      
      // round_off is a trick to enable integer division with ceil, even for negative numbers
      // We use a large offset, for the inner part not to become negative.
      const int round_off = ROUND_OFF;
      const int round_off_s1 = {{intStride}} * round_off;
      
      // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
      int xmin = (l - 3*{{intStride}} + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil (l - 3*{{intStride}}) / {{intStride}}
      int ymin = (m - 3*{{intStride}} + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil (l - 3*{{intStride}}) / {{intStride}}
      
      // Same here:
      int xmax = (l - 3*{{intStride}} + round_off_s1) / {{intStride}} - round_off; // floor (l - 3*{{intStride}}) / {{intStride}}
      int ymax = (m - 3*{{intStride}} + round_off_s1) / {{intStride}} - round_off; // floor (m - 3*{{intStride}}) / {{intStride}}
      
      float sum = 0;
      if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
        xmin = max(0,xmin);
        xmax = min(SIZE_3(gradOutput)-1,xmax);
        
        ymin = max(0,ymin);
        ymax = min(SIZE_2(gradOutput)-1,ymax);
        
        for (int p = -3; p <= 3; p++) {
          for (int o = -3; o <= 3; o++) {
            // Get rbot1 data:
            int s2o = {{intStride}} * o;
            int s2p = {{intStride}} * p;
            int idxbot1 = ((intSample * SIZE_1(rbot0) + (m+s2p)) * SIZE_2(rbot0) + (l+s2o)) * SIZE_3(rbot0) + n;
            float bot1tmp = rbot1[idxbot1]; // rbot1[l+s2o,m+s2p,n]
            
            // Index offset for gradOutput in following loops:
            int op = (p+3) * 7 + (o+3); // index[o,p]
            int idxopoffset = (intSample * SIZE_1(gradOutput) + op);
            
            for (int y = ymin; y <= ymax; y++) {
              for (int x = xmin; x <= xmax; x++) {
                int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
                sum += gradOutput[idxgradOutput] * bot1tmp;
              }
            }
          }
        }
      }
      const int sumelems = SIZE_1(gradOne);
      const int bot0index = ((n * SIZE_2(gradOne)) + (m-3*{{intStride}})) * SIZE_3(gradOne) + (l-3*{{intStride}});
      gradOne[bot0index + intSample*SIZE_1(gradOne)*SIZE_2(gradOne)*SIZE_3(gradOne)] = sum / (float)sumelems;
    } }
'''


kernel_Correlation_updateGradTwo = '''
    #define ROUND_OFF 50000

    extern "C" __global__ void kernel_Correlation_updateGradTwo(
      const int n,
      const int intSample,
      const float* rbot0,
      const float* rbot1,
      const float* gradOutput,
      float* gradOne,
      float* gradTwo
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
      int n = intIndex % SIZE_1(gradTwo); // channels
      int l = (intIndex / SIZE_1(gradTwo)) % SIZE_3(gradTwo) + 3*{{intStride}}; // w-pos
      int m = (intIndex / SIZE_1(gradTwo) / SIZE_3(gradTwo)) % SIZE_2(gradTwo) + 3*{{intStride}}; // h-pos
      
      // round_off is a trick to enable integer division with ceil, even for negative numbers
      // We use a large offset, for the inner part not to become negative.
      const int round_off = ROUND_OFF;
      const int round_off_s1 = {{intStride}} * round_off;
      
      float sum = 0;
      for (int p = -3; p <= 3; p++) {
        for (int o = -3; o <= 3; o++) {
          int s2o = {{intStride}} * o;
          int s2p = {{intStride}} * p;
          
          //Get X,Y ranges and clamp
          // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
          int xmin = (l - 3*{{intStride}} - s2o + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil (l - 3*{{intStride}} - s2o) / {{intStride}}
          int ymin = (m - 3*{{intStride}} - s2p + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil (l - 3*{{intStride}} - s2o) / {{intStride}}
          
          // Same here:
          int xmax = (l - 3*{{intStride}} - s2o + round_off_s1) / {{intStride}} - round_off; // floor (l - 3*{{intStride}} - s2o) / {{intStride}}
          int ymax = (m - 3*{{intStride}} - s2p + round_off_s1) / {{intStride}} - round_off; // floor (m - 3*{{intStride}} - s2p) / {{intStride}}
          
          if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
            xmin = max(0,xmin);
            xmax = min(SIZE_3(gradOutput)-1,xmax);
            
            ymin = max(0,ymin);
            ymax = min(SIZE_2(gradOutput)-1,ymax);
            
            // Get rbot0 data:
            int idxbot0 = ((intSample * SIZE_1(rbot0) + (m-s2p)) * SIZE_2(rbot0) + (l-s2o)) * SIZE_3(rbot0) + n;
            float bot0tmp = rbot0[idxbot0]; // rbot1[l+s2o,m+s2p,n]
            
            // Index offset for gradOutput in following loops:
            int op = (p+3) * 7 + (o+3); // index[o,p]
            int idxopoffset = (intSample * SIZE_1(gradOutput) + op);
            
            for (int y = ymin; y <= ymax; y++) {
              for (int x = xmin; x <= xmax; x++) {
                int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
                sum += gradOutput[idxgradOutput] * bot0tmp;
              }
            }
          }
        }
      }
      const int sumelems = SIZE_1(gradTwo);
      const int bot1index = ((n * SIZE_2(gradTwo)) + (m-3*{{intStride}})) * SIZE_3(gradTwo) + (l-3*{{intStride}});
      gradTwo[bot1index + intSample*SIZE_1(gradTwo)*SIZE_2(gradTwo)*SIZE_3(gradTwo)] = sum / (float)sumelems;
    } }
'''


def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction].replace(
        '{{intStride}}', str(objVariables['intStride']))
    while True:
        objMatch = re.search(r'(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)
        if objMatch is None:
            break
        intArg = int(objMatch.group(2))
        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()
        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg] if torch.is_tensor(
            intSizes[intArg]) == False else intSizes[intArg].item()))
    while True:
        objMatch = re.search(r'(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)
        if objMatch is None:
            break
        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')
        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')' for intArg in range(intArgs)]
        strKernel = strKernel.replace(objMatch.group(
            0), strTensor + '[' + str('+').join(strIndex) + ']')
    return strKernel


@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.RawKernel(strKernel, strFunction)


class _FunctionCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(self, one, two, intStride): # type: ignore
        rbot0 = one.new_zeros([one.shape[0], one.shape[2] + (6 * intStride),
                              one.shape[3] + (6 * intStride), one.shape[1]])
        rbot1 = one.new_zeros([one.shape[0], one.shape[2] + (6 * intStride),
                              one.shape[3] + (6 * intStride), one.shape[1]])
        self.intStride = intStride
        one = one.contiguous()
        assert (one.is_cuda == True)
        two = two.contiguous()
        assert (two.is_cuda == True)
        output = one.new_zeros([one.shape[0], 49, int(math.ceil(
            one.shape[2] / intStride)), int(math.ceil(one.shape[3] / intStride))])
        if one.is_cuda == True:
            n = one.shape[2] * one.shape[3]
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
                'intStride': self.intStride,
                'input': one,
                'output': rbot0
            }))(
                grid=tuple([int((n + 16 - 1) / 16),
                           one.shape[1], one.shape[0]]),
                block=tuple([16, 1, 1]),
                args=[cupy.int32(n), one.data_ptr(), rbot0.data_ptr()]
            )
            n = two.shape[2] * two.shape[3]
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
                'intStride': self.intStride,
                'input': two,
                'output': rbot1
            }))(
                grid=tuple([int((n + 16 - 1) / 16),
                           two.shape[1], two.shape[0]]),
                block=tuple([16, 1, 1]),
                args=[cupy.int32(n), two.data_ptr(), rbot1.data_ptr()]
            )
            n = output.shape[1] * output.shape[2] * output.shape[3]
            cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {
                'intStride': self.intStride,
                'rbot0': rbot0,
                'rbot1': rbot1,
                'top': output
            }))(
                grid=tuple(
                    [output.shape[3], output.shape[2], output.shape[0]]),
                block=tuple([32, 1, 1]),
                shared_mem=one.shape[1] * 4,
                args=[cupy.int32(n), rbot0.data_ptr(),
                      rbot1.data_ptr(), output.data_ptr()]
            )
        elif one.is_cuda == False:
            raise NotImplementedError()
        self.save_for_backward(one, two, rbot0, rbot1)
        return output

    @staticmethod
    def backward(self, gradOutput): # type: ignore
        one, two, rbot0, rbot1 = self.saved_tensors
        gradOutput = gradOutput.contiguous()
        assert (gradOutput.is_cuda == True)
        gradOne = one.new_zeros([one.shape[0], one.shape[1], one.shape[2],
                                one.shape[3]]) if self.needs_input_grad[0] == True else None
        gradTwo = one.new_zeros([one.shape[0], one.shape[1], one.shape[2],
                                one.shape[3]]) if self.needs_input_grad[1] == True else None
        if one.is_cuda == True:
            if gradOne is not None:
                for intSample in range(one.shape[0]):
                    n = one.shape[1] * one.shape[2] * one.shape[3]
                    cupy_launch('kernel_Correlation_updateGradOne', cupy_kernel('kernel_Correlation_updateGradOne', {
                        'intStride': self.intStride,
                        'rbot0': rbot0,
                        'rbot1': rbot1,
                        'gradOutput': gradOutput,
                        'gradOne': gradOne,
                        'gradTwo': None
                    }))(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[cupy.int32(n), intSample, rbot0.data_ptr(), rbot1.data_ptr(
                        ), gradOutput.data_ptr(), gradOne.data_ptr(), None]
                    )
            if gradTwo is not None:
                for intSample in range(one.shape[0]):
                    n = one.shape[1] * one.shape[2] * one.shape[3]
                    cupy_launch('kernel_Correlation_updateGradTwo', cupy_kernel('kernel_Correlation_updateGradTwo', {
                        'intStride': self.intStride,
                        'rbot0': rbot0,
                        'rbot1': rbot1,
                        'gradOutput': gradOutput,
                        'gradOne': None,
                        'gradTwo': gradTwo
                    }))(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[cupy.int32(n), intSample, rbot0.data_ptr(), rbot1.data_ptr(
                        ), gradOutput.data_ptr(), None, gradTwo.data_ptr()]
                    )
        elif one.is_cuda == False:
            raise NotImplementedError()
        return gradOne, gradTwo, None


def FunctionCorrelation(tenOne, tenTwo, intStride):
    return _FunctionCorrelation.apply(tenOne, tenTwo, intStride)


class ModuleCorrelation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tenOne, tenTwo, intStride):
        return _FunctionCorrelation.apply(tenOne, tenTwo, intStride)


def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()
    tenFlow = torch.cat([
        tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0)),
        tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0))], 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Features(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)
                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]

        class Matching(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()
                self.fltBackwarp = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()
                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )
                if intLevel == 6:
                    self.netUpflow = None
                elif intLevel != 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1, bias=False, groups=2)
                if intLevel >= 4:
                    self.netUpcorr = None
                elif intLevel < 4:
                    self.netUpcorr = torch.nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4, stride=2, padding=1, bias=False, groups=49)
                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
                )

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenFeaturesOne = self.netFeat(tenFeaturesOne)
                tenFeaturesTwo = self.netFeat(tenFeaturesTwo)
                if tenFlow is not None:
                    tenFlow = self.netUpflow(tenFlow) # type: ignore
                if tenFlow is not None:
                    tenFeaturesTwo = backwarp(tenInput=tenFeaturesTwo, tenFlow=tenFlow * self.fltBackwarp)
                if self.netUpcorr is None:
                    tenCorrelation = torch.nn.functional.leaky_relu(input=FunctionCorrelation(tenOne=tenFeaturesOne, tenTwo=tenFeaturesTwo, intStride=1), negative_slope=0.1, inplace=False)  # type: ignore
                elif self.netUpcorr is not None:
                    tenCorrelation = self.netUpcorr(torch.nn.functional.leaky_relu(input=FunctionCorrelation(tenOne=tenFeaturesOne, tenTwo=tenFeaturesTwo, intStride=2), negative_slope=0.1, inplace=False)) # type: ignore
                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(tenCorrelation) # type: ignore

        class Subpixel(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()
                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()
                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )
                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[0, 0, 130, 130, 194, 258, 386][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
                )

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenFeaturesOne = self.netFeat(tenFeaturesOne)
                tenFeaturesTwo = self.netFeat(tenFeaturesTwo)
                if tenFlow is not None:
                    tenFeaturesTwo = backwarp(tenInput=tenFeaturesTwo, tenFlow=tenFlow * self.fltBackward)
                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(torch.cat([tenFeaturesOne, tenFeaturesTwo, tenFlow], 1))

        class Regularization(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()
                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
                self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]
                if intLevel >= 5:
                    self.netFeat = torch.nn.Sequential()
                elif intLevel < 5:
                    self.netFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=[0, 0, 32, 64, 96, 128, 192][intLevel], out_channels=128, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )
                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[0, 0, 131, 131, 131, 131, 195][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                if intLevel >= 5:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel])
                    )
                elif intLevel < 5:
                    self.netDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], kernel_size=([0, 0, 7, 5, 5, 3, 3][intLevel], 1), stride=1, padding=([0, 0, 3, 2, 2, 1, 1][intLevel], 0)),
                        torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], kernel_size=(1, [0, 0, 7, 5, 5, 3, 3][intLevel]), stride=1, padding=(0, [0, 0, 3, 2, 2, 1, 1][intLevel]))
                    )
                self.netScaleX = torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
                self.netScaleY = torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenDifference = (tenOne - backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackward)).square().sum([1], True).sqrt().detach()
                tenDist = self.netDist(self.netMain(torch.cat([ tenDifference, tenFlow - tenFlow.mean([2, 3], True), self.netFeat(tenFeaturesOne) ], 1)))
                tenDist = tenDist.square().neg()
                tenDist = (tenDist - tenDist.max(1, True)[0]).exp()
                tenDivisor = tenDist.sum([1], True).reciprocal()
                tenScaleX = self.netScaleX(tenDist * torch.nn.functional.unfold(input=tenFlow[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor
                tenScaleY = self.netScaleY(tenDist * torch.nn.functional.unfold(input=tenFlow[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor
                return torch.cat([tenScaleX, tenScaleY], 1)

        self.netFeatures = Features()
        self.netMatching = torch.nn.ModuleList([Matching(intLevel) for intLevel in [ 2, 3, 4, 5, 6] ])
        self.netSubpixel = torch.nn.ModuleList([Subpixel(intLevel) for intLevel in [ 2, 3, 4, 5, 6] ])
        self.netRegularization = torch.nn.ModuleList([Regularization(intLevel) for intLevel in [ 2, 3, 4, 5, 6] ])
        self.load_state_dict({
            strKey.replace('module', 'net'): tenWeight
            for strKey, tenWeight in torch.hub.load_state_dict_from_url(
                url='http://content.sniklaus.com/github/pytorch-liteflownet/network-default.pytorch',
                file_name='liteflownet-default').items()
        })

    def forward(self, tenOne, tenTwo):
        tenOne[:, 0, :, :] = tenOne[:, 0, :, :] - 0.411618
        tenOne[:, 1, :, :] = tenOne[:, 1, :, :] - 0.434631
        tenOne[:, 2, :, :] = tenOne[:, 2, :, :] - 0.454253
        tenTwo[:, 0, :, :] = tenTwo[:, 0, :, :] - 0.410782
        tenTwo[:, 1, :, :] = tenTwo[:, 1, :, :] - 0.433645
        tenTwo[:, 2, :, :] = tenTwo[:, 2, :, :] - 0.452793
        tenFeaturesOne = self.netFeatures(tenOne)
        tenFeaturesTwo = self.netFeatures(tenTwo)
        tenOne = [tenOne]
        tenTwo = [tenTwo]
        for intLevel in [1, 2, 3, 4, 5]:
            tenOne.append(torch.nn.functional.interpolate(input=tenOne[-1], size=(tenFeaturesOne[intLevel].shape[2], tenFeaturesOne[intLevel].shape[3]), mode='bilinear', align_corners=False))
            tenTwo.append(torch.nn.functional.interpolate(input=tenTwo[-1], size=(tenFeaturesTwo[intLevel].shape[2], tenFeaturesTwo[intLevel].shape[3]), mode='bilinear', align_corners=False))
        tenFlow = None
        for intLevel in [-1, -2, -3, -4, -5]:
            tenFlow = self.netMatching[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel], tenFeaturesTwo[intLevel], tenFlow)
            tenFlow = self.netSubpixel[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel], tenFeaturesTwo[intLevel], tenFlow)
            tenFlow = self.netRegularization[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel], tenFeaturesTwo[intLevel], tenFlow)
        return tenFlow * 20.0 # type: ignore


def estimate(netNetwork, tenOne, tenTwo):

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()


def calc_optical_flow_liteflownet(prev_frame: Rgb, next_frame: Rgb) -> Flow:
    global netNetwork
    if netNetwork is None:
        netNetwork = Network().cuda().train(False)
    tenOne = torch.FloatTensor(numpy.ascontiguousarray(prev_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenTwo = torch.FloatTensor(numpy.ascontiguousarray(next_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenOutput = estimate(netNetwork, tenOne, tenTwo)
    flow = numpy.ascontiguousarray(numpy.array(tenOutput.numpy(force=True).transpose(1, 2, 0), numpy.float32))
    return cast(Flow, flow)
