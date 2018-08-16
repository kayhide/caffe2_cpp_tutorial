#include "caffe2/util/net.h"

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

namespace caffe2 {

// Helpers

void NetUtil::AddInput(const std::string input) {
  net.add_external_input(input);
}

void NetUtil::AddOutput(const std::string output) {
  net.add_external_output(output);
}

const std::string& NetUtil::Input(int i) {
  CAFFE_ENFORCE(net.external_input_size() != 0, net.name(),
                " doesn't have any exteral inputs");
  CAFFE_ENFORCE(net.external_input_size() > i, net.name(),
                " is missing exteral input ", i);
  return net.external_input(i);
}
const std::string& NetUtil::Output(int i) {
  CAFFE_ENFORCE(net.external_output_size() != 0, net.name(),
                " doesn't have any exteral outputs");
  CAFFE_ENFORCE(net.external_output_size() > i, net.name(),
                " is missing exteral output ", i);
  return net.external_output(i);
}

void NetUtil::SetName(const std::string name) { net.set_name(name); }

void NetUtil::SetType(const std::string type) { net.set_type(type); }

void NetUtil::SetEngineOps(const std::string engine) {
  for (auto& op : *net.mutable_op()) {
    op.set_engine(engine);
  }
}

void NetUtil::SetDeviceCUDA() {
#ifdef WITH_CUDA
  net.mutable_device_option()->set_device_type(CUDA);
#endif
}

}  // namespace caffe2
