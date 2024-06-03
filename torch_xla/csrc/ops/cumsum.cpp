#include "torch_xla/csrc/ops/cumsum.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace {

xla::XlaOp LowerCumSum(xla::XlaOp input, int64_t dim,
                       c10::optional<at::ScalarType> dtype, bool prepend) {
  XLA_CHECK(!prepend) << "prepend=True case not implemented for aten::cumsum";
  xla::XlaOp casted_input = CastToScalarType(input, dtype);
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(casted_input);
  xla::XlaOp init = XlaHelpers::ScalarValue<float>(
      0, input_shape.element_type(), casted_input.builder());
  xla::XlaComputation reducer =
      XlaHelpers::CreateAddComputation(input_shape.element_type());
  return BuildCumulativeComputation(casted_input, dim, reducer, init);
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           c10::optional<at::ScalarType> dtype, bool prepend) {
  XLA_CHECK(!prepend) << "prepend=True case not implemented for aten::cumsum";
  if (dtype) {
    return xla::ShapeUtil::ChangeElementType(
        GetXlaShape(input), MakeXlaPrimitiveType(*dtype, /*device=*/nullptr));
  }
  return GetXlaShape(input);
}

}  // namespace

CumSum::CumSum(const torch::lazy::Value& input, int64_t dim,
               c10::optional<at::ScalarType> dtype, bool prepend)
    : XlaNode(
          torch::lazy::OpKind(at::aten::cumsum), {input},
          [&]() { return NodeOutputShape(input, dtype, prepend); },
          /*num_outputs=*/1,
          torch::lazy::MHash(dim, torch::lazy::OptionalOr<int>(dtype, -1),
                             prepend)),
      dim_(dim),
      dtype_(dtype),
      prepend_(prepend) {}

torch::lazy::NodePtr CumSum::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<CumSum>(operands.at(0), dim_, dtype_, prepend_);
}

XlaOpVector CumSum::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerCumSum(input, dim_, dtype_, prepend_), loctx);
}

std::string CumSum::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  if (dtype_) {
    ss << ", dtype=" << *dtype_;
  }
  ss << ", prepend=" << prepend_;
  return ss.str();
}

}  // namespace torch_xla
