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
  // torch::Tensor result_tensor;

  int sr; 
  int block_size;
  int z_per_second;
  int prior_temp_size;
  int latent_size;
  std::atomic_bool loaded;
    
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

    this->z_per_second = this->block_size = this->latent_size = this->sr = this->prior_temp_size = -1;

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
        if (i.name == "_prior.previous_step" || i.name == "last_z") {
            // std::cout<<i.name<<std::endl;
            std::cout << i.value.sizes()[1] << std::endl;
            this->prior_temp_size = (int) i.value.sizes()[1];
        }
    } 

    if ((this->block_size<0) || 
        (this->latent_size<0)){
      std::cout << "model load failed" << std::endl;
      return;
    }
    if (this->prior_temp_size<0){
      std::cout << "WARNING: RAVE model in " << rave_model_file << " has no prior; RAVEPrior and RAVE with prior>0 will not function." << std::endl;
    }

    std::cout << "\tblock size: " << this->block_size << std::endl;
    std::cout << "\tlatent size: " << this->latent_size << std::endl;

    if (this->sr > 0){
      this->z_per_second = this->sr / this->block_size;
      std::cout << "\tsample rate: " << this->sr << std::endl;
    }

    c10::InferenceMode guard;
    inputs_rave.clear();
    inputs_rave.push_back(torch::ones({1,1,block_size}));

    //warmup
    this->model(inputs_rave);

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

  // for async
  void encode_decode(float* inBuffer, float* outBuffer){
    c10::InferenceMode guard;

    inputs_rave[0] = torch::from_blob(inBuffer, block_size)
      .reshape({1, 1, block_size});

    const auto result = this->model(inputs_rave).toTensor();

    auto data = result.data_ptr<float>();
    for (int i=0; i<block_size; ++i){
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

float sinc(float x){
  return x==0 ? 1.0 : std::sin(x*M_PI) / (x*M_PI);
}

// TODO: special case when rate_in == rate_out
class Resampler {
  public:
    long m_rate_in;
    long m_rate_out;
    int m_lanczos_n;
    long m_lanczos_rate;
    int m_filt_len;
    int m_head;
    //stored as # in samples
    long m_next_in;
    long m_last_in;
    // stored as # out samples 
    long m_next_out;
    long m_last_out;

    float delay;

    std::vector<float> m_values;
    std::vector<long> m_times;

    Resampler(){}

    Resampler(int rate_in, int rate_out, int lanczos_n){
      std::cout << "resampler: " << rate_in << " to " << rate_out << std::endl;

      m_rate_in = rate_in;
      m_rate_out = rate_out;
      m_lanczos_n = lanczos_n;
      m_lanczos_rate = std::min(rate_in, rate_out);

      // TODO: this should result in a m_filt_len=1 when lanczos_n==0 ...
      // could then just set to 0 if sample rates are the same 
      m_filt_len = int(std::ceil(
        rate_in / m_lanczos_rate * (m_lanczos_n + 1) * 2
      ));

      m_values = std::vector<float>(m_filt_len);
      m_times = std::vector<long>(m_filt_len);

      m_next_in = 0;
      m_last_in = -1;
      m_next_out = 0;
      m_last_out = -1;

      delay = float(m_lanczos_n) / m_lanczos_rate;
    }
    float get_dt(long t_in, long t_out){
      // std::cout << t_in << " " << t_out << std::endl;
      // std::cout << t_out * m_rate_in - t_in * m_rate_out << " " << m_rate_out * m_rate_in << std::endl;
      return 
        float(t_out * m_rate_in - t_in * m_rate_out)
        / (m_rate_out * m_rate_in);
    }
    float filter(float t){
      float t_center = t - delay; // in seconds
      float t_scale = t_center * m_lanczos_rate; // in samples at lanczos rate
      float w = 
        sinc(t_scale/m_lanczos_n) 
        * ((std::fabs(t_scale) <= m_lanczos_n) ? 1.0f : 0.0f);

      // std::cout << t << " " << delay << " " << t_center << " " << t_scale << " " << w << std::endl;

      return w * sinc(t_scale);
    }
    float read(){
      // DEBUG
      // m_last_out = m_next_out;m_next_out += 1; return m_values[0];

      float num = 0;
      float denom = 1e-15;
      for (int i=0; i<m_filt_len; i++){
        auto t = m_times[i];
        auto v = m_values[i];
        auto dt = get_dt(t, m_next_out);
        auto w = filter(dt);
        num += w * v;
        denom += w;
        // std::cout << "t " << t << "v " << v << "w " << w << std::endl;
      }
      m_last_out = m_next_out;
      m_next_out += 1;

      // std::cout << "read " << num << "/" << denom << std::endl;

      return num / denom;
    }
    void write(float x){
      //DEBUG
      // m_last_in = m_next_in; m_next_in += 1; m_values[0] = x; return;

      // std::cout << "x " << x << " m_head " << m_head << " m_filt_len " << m_filt_len << std::endl;
      m_values[m_head] = x;
      m_times[m_head] = m_next_in;

      m_last_in = m_next_in;
      m_next_in += 1;

      m_head += 1;
      m_head %= m_filt_len;
    }
    bool pending(){
      return 
        (m_last_in >= 0) && 
        (m_next_out * m_rate_in <= m_last_in * m_rate_out);
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

    float * modelInBuffer;
    float * modelOutBuffer;

    bool first_block_done;
    int filename_length;
    int ugen_inputs;
    int ugen_outputs;

    std::unique_ptr<std::thread> load_thread;
    std::unique_ptr<std::thread> compute_thread;
};

class AsyncRAVE : public RAVEBase {
  public:
    float delay; // estimated total delay in seconds
    Resampler res_in;
    Resampler res_out;
    long m_internal_samples; // count of total samples processed
    int m_processing_latency; // in model samples

    // override these
    const bool audio_in = true;
    const bool audio_out = true;
    AsyncRAVE();
    void next(int nSamples);
    // start the next block of processing
    // and read control rate inputs
    void dispatch(); 
    // finish the last block of processing
    // and write control rate outputs
    void join(); 

    // these work (or are unused) for all subclasses
    // dynamic alloc after loading / before running model
    void make_buffers(); 
    // ingest a single sample of audio input
    void write(float x); 
    // return a single sample of audio output
    float read();
    // read and write should be used together like so:
    float step(float x){write(x); return read();}
};

class RAVE : public RAVEBase {
  public:
    RAVE();
    void next(int nSamples);
    void make_buffers();
};
class RAVEEncoder : public RAVEBase {
  public:
    RAVEEncoder();
    void next(int nSamples);
    void make_buffers();
};
class RAVEDecoder : public RAVEBase {
  public:
    RAVEDecoder();
    // size_t ugen_inputs;
    void next(int nSamples);
    void make_buffers();
};
class RAVEPrior: public RAVEBase {
  public:
    RAVEPrior();
    void next(int nSamples);
    void make_buffers();
};

} // namespace RAVE
