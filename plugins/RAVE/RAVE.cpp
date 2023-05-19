// RAVE.cpp
// Victor Shepardson (victor.shepardson@gmail.com)

#include "RAVE.hpp"
#include "SC_PlugIn.hpp"

static InterfaceTable* ft;

namespace RAVE {

// auto RAVEBase::models = std::map<std::string, RAVEModel* >();

RAVEBase::RAVEBase() {
    inIdx = 0;
    outIdx = 0;
    first_block_done = false;

    filename_length = in0(0);
    std::cout<<filename_length<<std::endl;
    // char path[filename_length];
    auto path = std::string(filename_length, '!');
    for (int i=0; i<filename_length; i++){
        path[i] = static_cast<char>(in0(i+1));
    }

    model = new RAVEModel();
    std::cout << "loading: \"" << path << "\"" << std::endl;
    // model->load(path);
    this->compute_thread = std::make_unique<std::thread>(&RAVEModel::load, *model, path);

}

void RAVE::make_buffers() {
    inBuffer = (float*)RTAlloc(this->mWorld, model->block_size * sizeof(float));
    outBuffer = (float*)RTAlloc(this->mWorld, model->block_size * sizeof(float));
    this->ugen_outputs = 1;
}
RAVE::RAVE() : RAVEBase(){
    mCalcFunc = make_calc_function<RAVE, &RAVE::next>();
}

void RAVEPrior::make_buffers(){
    // currently unused but is freed in superclass destructor
    inBuffer = (float*)RTAlloc(this->mWorld, model->latent_size * sizeof(float));
    outBuffer = (float*)RTAlloc(this->mWorld, model->latent_size * sizeof(float));

    this->ugen_outputs = in0(filename_length+1);
    if (ugen_outputs != model->latent_size){
        std::cout << "WARNING: UGen outputs (" << ugen_outputs << ") do not match number of latent dimensions in model (" << model->latent_size << ")" << std::endl;
    }   
}
RAVEPrior::RAVEPrior() : RAVEBase(){
    mCalcFunc = make_calc_function<RAVEPrior, &RAVEPrior::next>();
}

void RAVEEncoder::make_buffers(){
    inBuffer = (float*)RTAlloc(this->mWorld, model->block_size * sizeof(float));
    outBuffer = (float*)RTAlloc(this->mWorld, model->latent_size * sizeof(float));
    this->ugen_outputs = in0(filename_length+1);
    if (ugen_outputs != model->latent_size){
        std::cout << "WARNING: UGen outputs (" << ugen_outputs << ") do not match number of latent dimensions in model (" << model->latent_size << ")" << std::endl;
    }   
}
RAVEEncoder::RAVEEncoder() : RAVEBase(){
    mCalcFunc = make_calc_function<RAVEEncoder, &RAVEEncoder::next>();
}

void RAVEDecoder::make_buffers(){
    inBuffer = (float*)RTAlloc(this->mWorld, model->latent_size * sizeof(float));
    outBuffer = (float*)RTAlloc(this->mWorld, model->block_size * sizeof(float));
    // filename len, *chars, inputs len, *inputs
    // number of inputs provided in synthdef
    this->ugen_inputs = in0(filename_length+1);
    this->ugen_outputs = 1;

    std::cout << 
        "RAVEDecoder latent size: " << model->latent_size << 
        "; found " << ugen_inputs << " inputs" << std::endl;
}
RAVEDecoder::RAVEDecoder() : RAVEBase(){
    mCalcFunc = make_calc_function<RAVEDecoder, &RAVEDecoder::next>();
}

RAVEBase::~RAVEBase() {
    RTFree(this->mWorld, inBuffer);
    RTFree(this->mWorld, outBuffer);
}

void RAVEBase::write_zeros_kr() {
    // std::cout<<"write zeros"<<std::endl;
    for (int j=0; j < this->ugen_outputs; ++j){
        out0(j) = 0;
    }
}
void RAVEBase::write_zeros_ar(int i) {
    // std::cout<<"write zeros"<<std::endl;
    for (int j=0; j < this->ugen_outputs; ++j){
        out(j)[i] = 0;
    }
}

void RAVE::next(int nSamples) {
    const float* input = in(filename_length+1);
    const float use_prior = in0(filename_length+2);
    const float temperature = in0(filename_length+3);
    float* output = out(0);

    if(!model->loaded){
        return;
    }
    else if(compute_thread->joinable()){
        compute_thread->join();
        make_buffers();
    }

    int model_block = model->block_size;
    int host_block = nSamples;

    // assume model_block and host_block are powers of two
    // handle case when model_block > host_block and the reverse
    int io_blocks = ceil(float(host_block) / model_block);
    int min_block = std::min(model_block, host_block);

    int hostInIdx = 0;
    int hostOutIdx = 0;

    for (int block = 0; block < io_blocks; ++block){

        for (int i = 0; i < min_block; ++i) {
            inBuffer[inIdx] = input[hostInIdx];
            hostInIdx++;
            inIdx++;
            if(inIdx == model_block){
                //process block
                if(use_prior && model->prior_temp_size>0){
                    model->prior_decode(temperature, outBuffer);
                } else {
                    model->encode_decode(inBuffer, outBuffer);
                }                
                outIdx = inIdx = 0;
                first_block_done = true;
            }
        }

        for (int i = 0; i < min_block; ++i) {
            if (first_block_done){
                if (outIdx >= model_block) {
                    std::cout<<"indexing error"<<std::endl;
                    outIdx = 0;
                }
                output[hostOutIdx] = outBuffer[outIdx];
                outIdx++;
                hostOutIdx++;
            }
            else {
                write_zeros_ar(i);
            }
        }
    }
}

void RAVEPrior::next(int nSamples) {
    const float temperature = in0(filename_length+2);

    if (!model->loaded) {
        write_zeros_kr();
        return;
    }
    else if(compute_thread->joinable()){
        compute_thread->join();
        make_buffers();
    }

    if (model->prior_temp_size<=0) {
        write_zeros_kr();
        return;
    }

    int model_block = model->block_size;
    int host_block = fullBufferSize();

    for (int i=0; i<host_block; ++i) {
        // just count samples, there is no audio input
        if(outIdx == 0){
            //process block
            model->prior(temperature, outBuffer);
            first_block_done = true;
        }
        outIdx++;
        if(outIdx == model_block){
            outIdx = 0;
        }
    }

    // write results to N kr outputs once per block
    if (first_block_done){
        for (int j=0; j<std::min(model->latent_size, this->ugen_outputs); ++j){
            out0(j) = outBuffer[j];
        }
    }
    else {
        write_zeros_kr();
    }
}

void RAVEEncoder::next(int nSamples) {
    const float* input = in(filename_length+2);

    if (!model->loaded) {
        write_zeros_kr();
        return;
    }
    else if(compute_thread->joinable()){
        compute_thread->join();
        make_buffers();
    }

    int model_block = model->block_size;
    int host_block = fullBufferSize();
     // assume model_block and host_block are powers of two
    // handle case when model_block > host_block and the reverse
    int io_blocks = ceil(float(host_block) / model_block);
    int min_block = std::min(model_block, host_block);

    int hostInIdx = 0;

    for (int block = 0; block < io_blocks; ++block){
        for (int i = 0; i < min_block; ++i) {
            inBuffer[inIdx] = input[hostInIdx];
            inIdx++;
            hostInIdx++;
            if(inIdx == model_block){
                //process block
                model->encode(inBuffer, outBuffer);

                inIdx = 0;
                first_block_done = true;
            }
        }
    }

    // write results to N kr outputs once per block
    if (first_block_done){
        for (int j=0; j < std::min(model->latent_size, this->ugen_outputs); ++j){
            out0(j) = outBuffer[j];
        }
    }
    else {
        write_zeros_kr();
    }
}

void RAVEDecoder::next(int nSamples) {
    float* output = out(0);

    if (!model->loaded){
        for (int i = 0; i < nSamples; ++i) {
            write_zeros_ar(i);
        }
        return;
    }
    else if(compute_thread->joinable()){
        compute_thread->join();
        make_buffers();
    }

    // read control-rate inputs once per frame
    int first_input = filename_length + 2;
    for (int j=0; j < model->latent_size; ++j){
        if (j<this->ugen_inputs){
            inBuffer[j] = in0(j + first_input);
        } else{
            inBuffer[j] = 0;
        }
    }

    int model_block = model->block_size;
    int host_block = nSamples;

    int hostOutIdx = 0;

    for (int i = 0; i < host_block; ++i) {
        if (outIdx==0) {
            model->decode(inBuffer, outBuffer);
        }
        output[hostOutIdx] = outBuffer[outIdx];
        hostOutIdx++;
        outIdx++;
        if (outIdx == model_block){
            outIdx = 0;
        }
    }

}

} // namespace RAVE

PluginLoad(RAVEUGens) {
    // Plugin magic
    ft = inTable;
    registerUnit<RAVE::RAVE>(ft, "RAVE", false);
    registerUnit<RAVE::RAVEPrior>(ft, "RAVEPrior", false);
    registerUnit<RAVE::RAVEEncoder>(ft, "RAVEEncoder", false);
    registerUnit<RAVE::RAVEDecoder>(ft, "RAVEDecoder", false);
}
