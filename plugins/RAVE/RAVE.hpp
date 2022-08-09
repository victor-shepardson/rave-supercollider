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

const int INPUT_SIZE = 2048;

struct RAVEModel {
  
  torch::jit::Module model;

  int sr;
  int decode_explosion;
  int z_per_second;
  int prior_temp_size;
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
    try {
        c10::InferenceMode guard;
        this->model = torch::jit::load(rave_model_file);
    }
    catch (const c10::Error& e) {
        std::cerr << e.what();
        std::cerr << e.msg();
        std::cerr << "error loading the model\n";
        return;
    }

    //
    // this->sr = 48000;
    // this->decode_explosion = 2048; 
    // this->z_per_second = (int)48000/2048;
    // this->prior_temp_size = 512;

    auto named_buffers = this->model.named_buffers();
    for (auto const& i: named_buffers) {
        if (i.name == "_rave.latent_size") {
            std::cout<<i.name<<std::endl;
            std::cout << i.value << std::endl;
        }
        if (i.name == "_rave.decode_params") {
            std::cout<<i.name<<std::endl;
            std::cout << i.value << std::endl;
            // why is this named `explosion`? appears to be the block size
            this->decode_explosion = i.value[1].item<int>();
        }
        if (i.name == "_rave.sampling_rate") {
            std::cout<<i.name<<std::endl;
            std::cout << i.value << std::endl;
            this->sr = i.value.item<int>();
        }
        if (i.name == "_prior.previous_step") {
            std::cout<<i.name<<std::endl;
            std::cout << i.value.sizes()[1] << std::endl;
            this->prior_temp_size = (int) i.value.sizes()[1];
        }
    } 
    this->z_per_second = this->sr / this->decode_explosion;

    c10::InferenceMode guard;
    inputs_rave.clear();
    inputs_rave.push_back(torch::ones({1,1,decode_explosion}));

    this->loaded = true;
  }
  
  torch::Tensor sample_from_prior (const float temperature) {
    c10::InferenceMode guard;

    inputs_rave[0] = torch::ones({1,1,1}) * temperature;
    const auto prior = this->model.get_method("prior")(inputs_rave).toTensor();

    inputs_rave[0] = prior;
    const auto y = this->model.get_method("decode")(inputs_rave).toTensor();

    return y.squeeze(0); // remove batch dim
  }

  torch::Tensor encode_decode (torch::Tensor input) {
    c10::InferenceMode guard;

    inputs_rave[0] = input;
    const auto y = this->model(inputs_rave).toTensor();

    return y.squeeze(0); // remove batch dim

  }

  torch::Tensor encode (torch::Tensor input) {
    c10::InferenceMode guard;

    inputs_rave[0] = input;
    const auto z = this->model.get_method("encode")(inputs_rave).toTensor();

    return z.squeeze(0); // remove batch dim
  }

  torch::Tensor decode (torch::Tensor latent) {
    c10::InferenceMode guard;

    inputs_rave[0] = latent;
    const auto y = this->model.get_method("decode")(inputs_rave).toTensor();

    return y.squeeze(0); // remove batch dim
  }

};

//unit command for passing in a model string
void load_model(struct Unit* unit, struct sc_msg_iter* args);

class RAVE : public SCUnit {

public:
    RAVE();
    ~RAVE();

    RAVEModel model;

private:
    float * inBuffer;
    size_t bufPtr;
    at::Tensor result;
    float * resultData;
    bool first_block;

    // Calc function
    void next(int nSamples);
};

} // namespace RAVE
