#ifndef KUIPER_INCLUDE_OP_RMSNORM_H
#define KUIPER_INCLUDE_OP_RMSNORM_H

#include "op/layer.h"

namespace op {
class RMSNormLaryer : public LayerParam {
public:
    explicit RMSNormLaryer(base::DeviceType device_type, int32_t dim);

    base::Status check() const override;

    base::Status forward() override;
private:
    int32_t dim_ = 0;
};
}  // namespace op

#endif  // KUIPER_INCLUDE_OP_RMSNORM_H