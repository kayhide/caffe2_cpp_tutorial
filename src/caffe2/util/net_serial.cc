#include "caffe2/util/net.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace caffe2 {

std::string NetUtil::Proto() {
  std::string s;
  google::protobuf::io::StringOutputStream stream(&s);
  google::protobuf::TextFormat::Print(net, &stream);
  return s;
}

void NetUtil::Print() {
  google::protobuf::io::OstreamOutputStream stream(&std::cout);
  google::protobuf::TextFormat::Print(net, &stream);
}

size_t NetUtil::Write(const std::string& path) const {
  WriteProtoToBinaryFile(net, path);
  return std::ifstream(path, std::ifstream::ate | std::ifstream::binary)
      .tellg();
}

size_t NetUtil::WriteText(const std::string& path) const {
  WriteProtoToTextFile(net, path);
  return std::ifstream(path, std::ifstream::ate | std::ifstream::binary)
      .tellg();
}

size_t NetUtil::Read(const std::string& path) {
  CAFFE_ENFORCE(ReadProtoFromFile(path.c_str(), &net));
  return std::ifstream(path, std::ifstream::ate | std::ifstream::binary)
      .tellg();
}

}  // namespace caffe2
