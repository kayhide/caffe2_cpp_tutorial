#include <caffe2/core/init.h>
#include <caffe2/core/blob.h>
#include <caffe2/core/net.h>
#include <caffe2/core/operator.h>

#include "caffe2/util/net.h"
#include "caffe2/util/blob.h"


CAFFE2_DEFINE_string(train_db, "res/mnist-train-nchw-leveldb",
                     "The given path to the training leveldb.");
CAFFE2_DEFINE_string(test_db, "res/mnist-test-nchw-leveldb",
                     "The given path to the testing leveldb.");
CAFFE2_DEFINE_int(iters, 100, "The of training runs.");
CAFFE2_DEFINE_int(test_runs, 50, "The of test runs.");
CAFFE2_DEFINE_bool(force_cpu, false, "Only use CPU, no CUDA.");
CAFFE2_DEFINE_bool(display, false, "Display graphical training info.");

namespace caffe2 {

  const std::string gradient_suffix("_grad");
  const std::string moment_suffix("_moment");
  const std::string meansq_suffix("_meansq");
  const std::string reader_suffix("_reader");
  const std::string init_net_suffix("_init_net.pb");
  const std::string predict_net_suffix("_predict_net.pb");
  const std::string model_info_suffix("_model_info.pb");
  const std::string init_name_suffix("_init");
  const std::string predict_name_suffix("_predict");

  const std::string iter_name("iter");
  const std::string lr_name("lr");
  const std::string one_name("one");
  const std::string loss_name("loss");
  const std::string label_name("label");
  const std::string xent_name("xent");
  const std::string accuracy_name("accuracy");

  class ModelUtil {
  public:
    ModelUtil(const std::string name)
      : init(NetUtil(init_net_)), predict(NetUtil(predict_net_))
    {
      init.SetName(name + init_name_suffix);
      predict.SetName(name + predict_name_suffix);
    }

    virtual ~ModelUtil() {}

  public:
    NetDef init_net_;
    NetDef predict_net_;
    NetUtil init;
    NetUtil predict;
  };

  void AddInput(ModelUtil& model,
                int batch_size,
                const std::string& db,
                const std::string& db_type) {
    // Setup database connection
    model.init.AddCreateDbOp("dbreader", db_type, db);
    model.predict.AddInput("dbreader");

    // >>> data_uint8, label = model.TensorProtosDBInput([], ["data_uint8",
    // "label"], batch_size=batch_size, db=db, db_type=db_type)
    model.predict.AddTensorProtosDbInputOp("dbreader", "data_uint8", "label",
                                           batch_size);

    // >>> data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    model.predict.AddCastOp("data_uint8", "data", TensorProto_DataType_FLOAT);

    // >>> data = model.Scale(data, data, scale=float(1./256))
    model.predict.AddScaleOp("data", "data", 1.f / 256);

    // >>> data = model.StopGradient(data, data)
    model.predict.AddStopGradientOp("data");
  }

  void AddLeNetInit(ModelUtil& model) {
    model.init.AddXavierFillOp({20, 1, 5, 5}, "conv1_w");
    model.init.AddConstantFillOp({20}, "conv1_b");
    model.init.AddXavierFillOp({50, 20, 5, 5}, "conv2_w");
    model.init.AddConstantFillOp({50}, "conv2_b");
    model.init.AddXavierFillOp({500, 50 * 4 * 4}, "fc3_w");
    model.init.AddConstantFillOp({500}, "fc3_b");
    model.init.AddXavierFillOp({10, 500}, "fc4_w");
    model.init.AddConstantFillOp({10}, "fc4_b");
  }

  void AddLeNet(ModelUtil& model) {
    model.predict.AddInput("conv1_w");
    model.predict.AddInput("conv1_b");
    model.predict.AddConvOp("data", "conv1_w", "conv1_b", "conv1", 1, 0, 5);

    model.predict.AddMaxPoolOp("conv1", "pool1", 2, 0, 2);

    model.predict.AddInput("conv2_w");
    model.predict.AddInput("conv2_b");
    model.predict.AddConvOp("pool1", "conv2_w", "conv2_b", "conv2", 1, 0, 5);

    model.predict.AddMaxPoolOp("conv2", "pool2", 2, 0, 2);

    model.predict.AddInput("fc3_w");
    model.predict.AddInput("fc3_b");
    model.predict.AddFcOp("pool2", "fc3_w", "fc3_b", "fc3");

    model.predict.AddReluOp("fc3", "relu3");

    model.predict.AddInput("fc4_w");
    model.predict.AddInput("fc4_b");
    model.predict.AddFcOp("relu3", "fc4_w", "fc4_b", "fc4");

    model.predict.AddSoftmaxOp("fc4", "softmax");
  }

  void AddTrainer(ModelUtil& model) {
    model.predict.AddLabelCrossEntropyOp("softmax", "label", "xent");

    model.predict.AddAveragedLossOp("xent", "loss");

    model.predict.AddConstantFillWithOp(1.0, "loss", "loss_grad");
    model.predict.AddGradientOps();

    // >>> LR = model.LearningRate(ITER, "LR", base_lr=-0.1, policy="step",
    // stepsize=1, gamma=0.999 )
    model.init.AddConstantFillOp({1}, (int64_t)0, iter_name)
      ->mutable_device_option()
      ->set_device_type(CPU);
    model.predict.AddInput(iter_name);
    model.predict.AddIterOp(iter_name);
    model.predict.AddLearningRateOp("iter", "LR", 0.1);

    // >>> ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1],
    // value=1.0)
    model.init.AddConstantFillOp({1}, 1.f, "ONE");
    model.predict.AddInput("ONE");

    // >>> for param in model.params:
    for (auto param : model.predict.CollectParams()) {
      // >>> param_grad = model.param_to_grad[param]
      // >>> model.WeightedSum([param, ONE, param_grad, LR], param)
      model.predict.AddWeightedSumOp({param, "ONE", param + "_grad", "LR"}, param);
    }
  }

  void AddAccuracy(ModelUtil& model) {
    model.predict.AddAccuracyOp("softmax", "label", "accuracy");
  }

  void Save(ModelUtil& model, const std::string dir) {
    {
      auto file = dir + "/" + model.init.net.name() + ".pbtxt";
      model.init.WriteText(file);
      std::cout << "create: " << file << std::endl;
    }
    {
      auto file = dir + "/" + model.init.net.name() + ".pb";
      model.init.Write(file);
      std::cout << "create: " << file << std::endl;
    }
    {
      auto file = dir + "/" + model.predict.net.name() + ".pbtxt";
      model.predict.WriteText(file);
      std::cout << "create: " << file << std::endl;
    }
    {
      auto file = dir + "/" + model.predict.net.name() + ".pb";
      model.predict.Write(file);
      std::cout << "create: " << file << std::endl;
    }
  }

  ModelUtil Load(const std::string dir, const std::string name) {
    ModelUtil model(name);
    {
      auto file = dir + "/" + model.init.net.name() + ".pb";
      model.init.Read(file);
      std::cout << "load: " << file << std::endl;
    }
    {
      auto file = dir + "/" + model.predict.net.name() + ".pb";
      model.predict.Read(file);
      std::cout << "load: " << file << std::endl;
    }
    return model;
  }



  void run () {

    std::cout << std::endl;
    std::cout << "## Caffe2 MNIST Tutorial ##" << std::endl;
    std::cout << "https://caffe2.ai/docs/tutorial-MNIST.html" << std::endl;
    std::cout << std::endl;

    if (!std::ifstream(FLAGS_train_db).good() ||
        !std::ifstream(FLAGS_test_db).good()) {
      std::cerr << "error: MNIST database missing: "
                << (std::ifstream(FLAGS_train_db).good() ? FLAGS_test_db
                    : FLAGS_train_db)
                << std::endl;
      std::cerr << "Make sure to first run ./script/download_resource.sh"
                << std::endl;
      return;
    }

    std::cout << "train-db: " << FLAGS_train_db << std::endl;
    std::cout << "test-db: " << FLAGS_test_db << std::endl;
    std::cout << "iters: " << FLAGS_iters << std::endl;
    std::cout << "test-runs: " << FLAGS_test_runs << std::endl;
    std::cout << "force-cpu: " << (FLAGS_force_cpu ? "true" : "false")
              << std::endl;
    std::cout << "display: " << (FLAGS_display ? "true" : "false") << std::endl;

    // >>> from caffe2.python import core, cnn, net_drawer, workspace, visualize,
    // brew
    // >>> workspace.ResetWorkspace(root_folder)
    Workspace workspace("tmp");

    // >>> train_model = model_helper.ModelHelper(name="mnist_train",
    // arg_scope={"order": "NCHW"})
    ModelUtil train("mnist_train");
    {
      const int batch_size = 64;
      AddInput(train, batch_size, FLAGS_train_db, "leveldb");
      AddLeNet(train);
      AddLeNetInit(train);
      AddTrainer(train);
      AddAccuracy(train);

      workspace.RunNetOnce(train.init.net);
      workspace.CreateNet(train.predict.net);

      std::cout << std::endl;
      std::cout << "training.." << std::endl;

      // >>> for i in range(total_iters):
      for (auto i = 1; i <= FLAGS_iters; i++) {
        workspace.RunNet(train.predict.net.name());

        if (i % 10 == 0) {
          auto accuracy =
            BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
          auto loss = BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
          std::cout << "step: " << i << " loss: " << loss
                    << " accuracy: " << accuracy << std::endl;
        }
      }
    }

    ModelUtil test("mnist_test");
    {
      const int batch_size = 100;
      AddInput(test, batch_size, FLAGS_test_db, "leveldb");
      AddLeNet(test);
      AddAccuracy(test);

      workspace.RunNetOnce(test.init.net);
      workspace.CreateNet(test.predict.net);

      std::cout << std::endl;
      std::cout << "testing.." << std::endl;
      for (auto i = 1; i <= FLAGS_test_runs; i++) {
        workspace.RunNet(test.predict.net.name());

        if (i % 10 == 0) {
          auto accuracy =
            BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
          std::cout << "step: " << i << " accuracy: " << accuracy << std::endl;
        }
      }
    }

    ModelUtil deploy("mnist");
    {
      deploy.predict.AddInput("data");
      deploy.predict.AddOutput("softmax");
      AddLeNet(deploy);

      for (auto &param : deploy.predict.net.external_input()) {
        auto &tensor = BlobUtil(*workspace.GetBlob(param)).Get();
        auto op = deploy.init.net.add_op();
        op->set_type("GivenTensorFill");
        auto arg1 = op->add_arg();
        arg1->set_name("shape");
        for (auto d : tensor.dims()) {
          arg1->add_ints(d);
        }
        auto arg2 = op->add_arg();
        arg2->set_name("values");
        auto data = tensor.data<float>();
        for (auto i = 0; i < tensor.size(); i++) {
          arg2->add_floats(data[i]);
        }
        op->add_output(param);
      }
    }

    {
      std::cout << std::endl;
      std::cout << "saving model.." << std::endl;

      Save(train, "tmp");
      Save(test, "tmp");
      Save(deploy, "tmp");
    }
  }
}


int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  // caffe2::predict_example();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
