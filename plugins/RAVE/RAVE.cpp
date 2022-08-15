// RAVE.cpp
// Victor Shepardson (victor.shepardson@gmail.com)

#include "RAVE.hpp"
#include "SC_PlugIn.hpp"

static InterfaceTable* ft;

namespace RAVE {

auto RAVEBase::models = std::map<std::string, RAVEModel* >();

RAVEBase::RAVEBase() {
    bufPtr = 0;
    first_block_done = false;

    filename_length = in0(0);
    std::cout<<filename_length<<std::endl;
    // char path[filename_length];
    auto path = std::string(filename_length, '!');
    for (int i=0; i<filename_length; i++){
        path[i] = static_cast<char>(in0(i+1));
    }

    auto kv = models.find(path);
    if (kv==models.end()){
        model = new RAVEModel();
        std::cout << "loading: \"" << path << "\"" << std::endl;
        model->load(path);
        models.insert({path, model});
    } else {
        model = kv->second;
        std::cout << "found \"" << path << "\" already loaded" << std::endl;
    }
}

// TODO: how to avoid the hardcoded values here?
RAVE::RAVE() : RAVEBase(){
    inBuffer = (float*)RTAlloc(this->mWorld, model->block_size * sizeof(float));
    outBuffer = (float*)RTAlloc(this->mWorld, model->block_size * sizeof(float));
    mCalcFunc = make_calc_function<RAVE, &RAVE::next>();
}

RAVEPrior::RAVEPrior() : RAVEBase(){
    // currently unused but is freed in superclass destructor
    inBuffer = (float*)RTAlloc(this->mWorld, model->latent_size * sizeof(float));
    outBuffer = (float*)RTAlloc(this->mWorld, model->latent_size * sizeof(float));
    mCalcFunc = make_calc_function<RAVEPrior, &RAVEPrior::next>();
}

RAVEEncoder::RAVEEncoder() : RAVEBase(){
    inBuffer = (float*)RTAlloc(this->mWorld, model->block_size * sizeof(float));
    outBuffer = (float*)RTAlloc(this->mWorld, model->latent_size * sizeof(float));
    mCalcFunc = make_calc_function<RAVEEncoder, &RAVEEncoder::next>();

    // std::cout << 
        // "RAVEEncoder latent size: " << model->latent_size << 
        // "; creating " << model->latent_size << " outputs" << std::endl;
}

RAVEDecoder::RAVEDecoder() : RAVEBase(){
    inBuffer = (float*)RTAlloc(this->mWorld, model->latent_size * sizeof(float));
    outBuffer = (float*)RTAlloc(this->mWorld, model->block_size * sizeof(float));
    mCalcFunc = make_calc_function<RAVEDecoder, &RAVEDecoder::next>();

    // filename len, *chars, inputs len, *inputs

    // number of inputs provided in synthdef
    this->ugen_inputs = in0(filename_length+1);

    std::cout << 
        "RAVEDecoder latent size: " << model->latent_size << 
        "; found " << ugen_inputs << " inputs" << std::endl;
}

RAVEBase::~RAVEBase() {
    RTFree(this->mWorld, inBuffer);
    RTFree(this->mWorld, outBuffer);
}

void RAVEBase::write_zeros_kr() {
    // std::cout<<"write zeros"<<std::endl;
    for (int j=0; j < model->latent_size; ++j){
        out0(j) = 0;
    }
}

void RAVE::next(int nSamples) {
    const float* input = in(filename_length+1);
    const float use_prior = in0(filename_length+2);
    const float temperature = in0(filename_length+3);

    float* output = out(0);

    for (int i = 0; i < nSamples; ++i) {
        if (!model->loaded) {
            output[i] = 0;
            continue;
        }

        inBuffer[bufPtr] = input[i];
        bufPtr++;
        if(bufPtr == model->block_size){
            //process block
            if(use_prior){
                model->prior_decode(temperature, outBuffer);
            } else {
                model->encode_decode(inBuffer, outBuffer);
            }

            bufPtr = 0;
            first_block_done = true;
        }

        if (first_block_done){
            output[i] = outBuffer[bufPtr];
        }
        else {
            output[i] = 0;
        }

    }
}

void RAVEPrior::next(int nSamples) {
    const float temperature = in0(filename_length+1);

    if (!model->loaded) {
        write_zeros_kr();
        return;
    }

    for (int i=0; i<fullBufferSize(); ++i) {
        // just count samples, there is no audio input
        bufPtr++;
        if(bufPtr == model->block_size){
            //process block
            model->prior(temperature, outBuffer);

            bufPtr = 0;
            first_block_done = true;
        }
    }

    // write results to N kr outputs once per block
    if (first_block_done){
        for (int j=0; j<model->latent_size; ++j){
            out0(j) = outBuffer[j];
        }
    }
    else {
        write_zeros_kr();
    }
}

void RAVEEncoder::next(int nSamples) {
    const float* input = in(filename_length+1);

    if (!model->loaded) {
        write_zeros_kr();
        return;
    }

    for (int i = 0; i < fullBufferSize(); ++i) {

        inBuffer[bufPtr] = input[i];
        bufPtr++;
        if(bufPtr == model->block_size){
            //process block
            model->encode(inBuffer, outBuffer);

            bufPtr = 0;
            first_block_done = true;
        }
    }

    // write results to N kr outputs once per block
    if (first_block_done){
        for (int j=0; j < model->latent_size; ++j){
            out0(j) = outBuffer[j];
        }
    }
    else {
        write_zeros_kr();
    }
}

void RAVEDecoder::next(int nSamples) {
    float* output = out(0);

    for (int i = 0; i < nSamples; ++i) {
        if (!model->loaded) {
            output[i] = 0;
            continue;
        }

        bufPtr++;
        if(bufPtr == model->block_size){
            // read only up to latent_size inputs,
            // or zero any extra latents if there are fewer inputs
            for (int j=0; j < model->latent_size; ++j){
                if (j<this->ugen_inputs){
                    inBuffer[j] = in0(j + filename_length + 2);
                } else{
                    inBuffer[j] = 0;
                }
            }
            //process block
            model->decode(inBuffer, outBuffer);

            bufPtr = 0;
            first_block_done = true;
        }

        if (first_block_done){
            output[i] = outBuffer[bufPtr];
        }
        else {
            output[i] = 0;
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
