// RAVE.hpp
// Victor Shepardson (victor.shepardson@gmail.com)

// parts of this file are adapted from code by Andrew Fyfe and Bogdan Teleaga
// licensed under GPLv3
// https://github.com/Fyfe93/RAVE-audition/blob/main/Source/Rave.h

#pragma once

// note: preprocessor wants torch included first
#include <torch/script.h>

#include "SC_PlugIn.hpp"

namespace RAVE {

// RAVEModel encapsulates the libtorch parts
struct RAVEModel {

  torch::jit::Module model;

  int sr;
  int block_size;
  int z_per_second;
  int prior_temp_size;
  int latent_size;
  bool loaded;
    
  std::vector<torch::jit::IValue> inputs_rave;

  RAVEModel() {
//    torch::init_num_threads();
//    unsigned int num_threads = std::thread::hardware_concurrency();
//
//    if (num_threads > 0) {
//        torch::set_num_threads((int)num_threads);
//        torch::set_num_interop_threads((int)num_threads);
//    } else {
//        torch::set_num_threads(4);
//        torch::set_num_interop_threads(4);
//    }
    this->loaded=false;
    torch::jit::getProfilingMode() = false;
    c10::InferenceMode guard;
    torch::jit::setGraphExecutorOptimize(true);
    }
    
  void load(const std::string& rave_model_file) {
    // std::cout << "\"" <<rave_model_file << "\"" <<std::endl;
    try {
        c10::InferenceMode guard;
        this->model = torch::jit::load(rave_model_file);
    }
    catch (const c10::Error& e) {
      // why no error when filename is bad?
        std::cout << e.what();
        std::cout << e.msg();
        std::cout << "error loading the model\n";
        return;
    }

    // support for Neutone models
    if (this->model.hasattr("model")){
      this->model = this->model.attr("model").toModule();
    }

    this->block_size = this->latent_size = this->sr = this->prior_temp_size = -1;

    auto named_buffers = this->model.named_buffers();
    for (auto const& i: named_buffers) {
        // std::cout<<i.name<<std::endl;
        
        // if ((i.name == "_rave.latent_size") || (i.name == "latent_size")) {
        //     std::cout<<i.name<<std::endl;
        //     std::cout << i.value << std::endl;
        //     this -> latent_size = i.value.item<int>();
        // }
        if (
          (i.name == "_rave.decode_params") 
          || (i.name == "decode_params")
          ) {
            // std::cout<<i.name<<std::endl;
            // std::cout << i.value << std::endl;
            this->block_size = i.value[1].item<int>();
            this->latent_size = i.value[0].item<int>();
        }
        if (
          (i.name == "_rave.sampling_rate") 
          || (i.name == "sampling_rate")
          ) {
            // std::cout<<i.name<<std::endl;
            // std::cout << i.value << std::endl;
            this->sr = i.value.item<int>();
        }
        if (i.name == "_prior.previous_step") {
            // std::cout<<i.name<<std::endl;
            std::cout << i.value.sizes()[1] << std::endl;
            this->prior_temp_size = (int) i.value.sizes()[1];
        }
    } 
    this->z_per_second = this->sr / this->block_size;

    if ((this->block_size<0) || 
        (this->latent_size<0) || 
        (this->sr<0)){
      std::cout << "model load failed" << std::endl;
      return;
    }
    if (this->prior_temp_size<0){
      std::cout << "WARNING: RAVE model in " << rave_model_file << " has no prior; RAVEPrior and RAVE with prior>0 will not function." << std::endl;
    }

    std::cout << "\tblock size: " << this->block_size << std::endl;
    std::cout << "\tlatent size: " << this->latent_size << std::endl;
    std::cout << "\tsample rate: " << this->sr << std::endl;

    c10::InferenceMode guard;
    inputs_rave.clear();
    inputs_rave.push_back(torch::ones({1,1,block_size}));

    this->loaded = true;
  }


  void prior_decode (const float temperature, float* outBuffer) {
    c10::InferenceMode guard;

    inputs_rave[0] = torch::ones({1, 1, 1}) * temperature;
    const auto prior = this->model.get_method("prior")(
      inputs_rave).toTensor();

    inputs_rave[0] = prior;
    const auto y = this->model.get_method("decode")(
      inputs_rave).toTensor().contiguous();

    auto data = y.data_ptr<float>();
    for (int i=0; i<block_size; i++){
      outBuffer[i] = data[i];
    }
  }

  void encode_decode (float* input, float* outBuffer) {
    c10::InferenceMode guard;

    inputs_rave[0] = torch::from_blob(
      input, block_size).reshape({1, 1, block_size});

    const auto y = this->model(inputs_rave).toTensor();

    auto data = y.data_ptr<float>();
    for (int i=0; i<block_size; i++){
      outBuffer[i] = data[i];
    }  
  }

  // TODO: version of prior which consumes last frame
  // is this possible without custom RAVE export?
  void prior (const float temperature, float* outBuffer) {
    c10::InferenceMode guard;

    inputs_rave[0] = torch::ones({1, 1, 1}) * temperature;
    const auto z = this->model.get_method("prior")(
      inputs_rave).toTensor();

    auto data = z.data_ptr<float>();
    for (int i=0; i<latent_size; i++){
      outBuffer[i] = data[i];
    }
  }

  void encode (float* input, float* outBuffer) {
    c10::InferenceMode guard;

    inputs_rave[0] = torch::from_blob(
      input, block_size).reshape({1, 1, block_size});

    const auto z = this->model.get_method("encode")(
      inputs_rave).toTensor();

    auto data = z.data_ptr<float>();
    for (int i=0; i<latent_size; i++){
      outBuffer[i] = data[i];
    }
  }

  void decode (float* latent, float* outBuffer) {
    c10::InferenceMode guard;

    inputs_rave[0] = torch::from_blob(
      latent, latent_size).reshape({1, latent_size, 1});

    const auto y = this->model.get_method("decode")(
      inputs_rave).toTensor();

    auto data = y.data_ptr<float>();
    for (int i=0; i<block_size; i++){
      outBuffer[i] = data[i];
    }  
  }

};

// RAVEBase has common parts of Encoder, Decoder, Prior functions
class RAVEBase : public SCUnit {

public:
    RAVEBase();
    ~RAVEBase();

    void write_zeros_kr();
    void write_zeros_ar(int i);

    RAVEModel * model;
    static std::map<std::string, RAVEModel* > models;

    float * inBuffer; // allocated in subclass constructor
    size_t inIdx;

    float * outBuffer; // allocated in subclass constructor
    size_t outIdx;

    bool first_block_done;
    int filename_length;
    int ugen_inputs;
    int ugen_outputs;

};

class RAVE : public RAVEBase {
  public:
    RAVE();
    void next(int nSamples);
};
class RAVEEncoder : public RAVEBase {
  public:
    RAVEEncoder();
    void next(int nSamples);
};
class RAVEDecoder : public RAVEBase {
  public:
    RAVEDecoder();
    size_t ugen_inputs;
    void next(int nSamples);
};
class RAVEPrior: public RAVEBase {
  public:
    RAVEPrior();
    void next(int nSamples);
};

} // namespace RAVE
