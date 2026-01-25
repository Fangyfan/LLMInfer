#include "softmax_kernel.h"
#include <algorithm>
#include <armadillo>

namespace kernel {
void softmax_kernel_cpu(const tensor::Tensor& input, void* stream) {
    UNUSED(stream);
    CHECK(!input.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCPU);

    int32_t size = input.size();
    float* input_ptr = const_cast<float*>(input.ptr<float>());
    float max_val = *std::max_element(input_ptr, input_ptr + size);

    arma::fvec in(input_ptr, size, false, true);
    in = arma::exp(in - max_val);

    float sum_exp = arma::sum(in);
    in /= sum_exp;
}
}  // namespace kernel