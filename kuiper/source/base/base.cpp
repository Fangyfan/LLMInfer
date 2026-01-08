#include "base/base.h"

namespace base {

Status::Status(StatusCode code, std::string err_msg) : code_(code), err_msg_(err_msg) {}

Status& Status::operator=(StatusCode code) {
    code_ = code;
    return *this;
}

bool Status::operator==(StatusCode code) const {
    return code_ == code;
}

bool Status::operator!=(StatusCode code) const {
    return code_ != code;
}

Status::operator int() const {
    return int(code_);
}

Status::operator bool() const {
    return code_ == StatusCode::Success;
}

StatusCode Status::get_err_code() const {
    return code_;
}

const std::string& Status::get_err_msg() const {
    return err_msg_;
}

void Status::set_err_msg(const std::string& err_msg) {
    err_msg_ = err_msg;
}

namespace error {
Status success(const std::string& err_msg) {
    return Status { StatusCode::Success, err_msg };
}

Status function_not_implement(const std::string& err_msg) {
    return Status { StatusCode::FunctionUnImplement, err_msg };
}

Status path_not_valid(const std::string& err_msg) {
    return Status { StatusCode::PathNotValid, err_msg };
}

Status model_parse_error(const std::string& err_msg) {
    return Status { StatusCode::ModelParseError, err_msg };
}

Status internal_error(const std::string& err_msg) {
    return Status { StatusCode::InternalError, err_msg };
}

Status key_has_exits(const std::string& err_msg) {
    return Status { StatusCode::KeyValueHasExist, err_msg };
}

Status invalid_argument(const std::string& err_msg) {
    return Status { StatusCode::InvalidArgument, err_msg };
}
}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& status) {
    os << status.get_err_msg();
    return os;
}

}  // namespace base