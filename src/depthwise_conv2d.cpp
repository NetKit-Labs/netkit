#include "depthwise_conv2d.hpp"
#include "active_kernel.hpp"

bool DepthwiseConv2D::forward(const Tensor& input, Tensor& output, NetkitKernelActivation fuse_activation)
{
    const int pad_h_end = this->pad_h_end >= 0 ? this->pad_h_end : this->pad_h;
    const int pad_w_end = this->pad_w_end >= 0 ? this->pad_w_end : this->pad_w;
    return Kernels::DepthwiseConv2dForward(input,
                                           weights,
                                           bias,
                                           kernel_h,
                                           kernel_w,
                                           stride,
                                           pad_h,
                                           pad_w,
                                           pad_h_end,
                                           pad_w_end,
                                           channels,
                                           fuse_activation,
                                           output);
}
