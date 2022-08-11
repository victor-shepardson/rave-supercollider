// RAVE.cpp
// Victor Shepardson (victor.shepardson@gmail.com)

#include "RAVE.hpp"
#include "SC_PlugIn.hpp"

static InterfaceTable* ft;

namespace RAVE {

// RAVEModel RAVEBase::model = RAVEModel();

auto RAVEBase::models = std::map<std::string, RAVEModel>();

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
        model = RAVEModel();
        std::cout << "loading: \"" << path << "\"" << std::endl;
        model.load(path);
        models.insert({path, model});
    } else {
        model = kv->second;
        std::cout << "found \"" << path << "\" already loaded" << std::endl;
    }
}

// TODO: how to avoid the hardcoded values here?
RAVE::RAVE() : RAVEBase(){
    inBuffer = (float*)RTAlloc(this->mWorld, INPUT_SIZE * sizeof(float));
    mCalcFunc = make_calc_function<RAVE, &RAVE::next>();
}

RAVEEncoder::RAVEEncoder() : RAVEBase(){
    inBuffer = (float*)RTAlloc(this->mWorld, INPUT_SIZE * sizeof(float));
    mCalcFunc = make_calc_function<RAVEEncoder, &RAVEEncoder::next>();
    // next(INPUT_SIZE);
}

RAVEDecoder::RAVEDecoder() : RAVEBase(){
    inBuffer = (float*)RTAlloc(this->mWorld, LATENT_SIZE * sizeof(float));
    mCalcFunc = make_calc_function<RAVEDecoder, &RAVEDecoder::next>();
    // next(INPUT_SIZE);

    // filename len, *chars, inputs len, *inputs
    ugen_inputs = in0(filename_length+1);

    std::cout << 
        "model latent size: " << model.latent_size << 
        "; found " << ugen_inputs << " inputs" << std::endl;
}

RAVEBase::~RAVEBase() {
    RTFree(this->mWorld, inBuffer);
}

void RAVE::next(int nSamples) {
    const float* input = in(filename_length+1);
    const float use_prior = in0(filename_length+2);
    const float temperature = in0(filename_length+3);

    float* outbuf = out(0);

    for (int i = 0; i < nSamples; ++i) {
        if (!model.loaded) {
            outbuf[i] = 0;
            continue;
        }

        inBuffer[bufPtr] = input[i];
        bufPtr++;
        if(bufPtr == INPUT_SIZE){
            //process block
            at::Tensor frame = torch::from_blob(inBuffer, model.block_size);
            if(use_prior){
                result = model.sample_from_prior(temperature);
            } else {
                frame = torch::reshape(frame, {1,1,model.block_size});
                result = model.encode_decode(frame);
            }
            resultData = result.data_ptr<float>();

            bufPtr = 0;
            first_block_done = true;
        }

        if (first_block_done){
            outbuf[i] = resultData[bufPtr];
        }
        else {
            outbuf[i] = 0;
        }

    }
}

void RAVEEncoder::next(int nSamples) {
    const float* input = in(filename_length+1);

    // should write zeros here but don't know how many outputs before model is loaded
    // leaving this since we want to change how model loading works anyway...
    if (!model.loaded) return;

    for (int i = 0; i < fullBufferSize(); ++i) {

        inBuffer[bufPtr] = input[i];
        bufPtr++;
        if(bufPtr == INPUT_SIZE){
            //process block
            at::Tensor frame = torch::from_blob(inBuffer, model.block_size);
            frame = torch::reshape(frame, {1,1,model.block_size});
            result = model.encode(frame);
            resultData = result.data_ptr<float>();

            bufPtr = 0;
            first_block_done = true;
        }
    }

    // write results to N kr outputs once per block
    if (first_block_done){
        for (int j=0; j < model.latent_size; j++){
            out0(j) = resultData[j];
        }
    }
    else {
        for (int j=0; j < model.latent_size; j++){
            out0(j) = 0;
        }
    }
}

void RAVEDecoder::next(int nSamples) {
    float* outbuf = out(0);

    for (int i = 0; i < nSamples; ++i) {
        if (!model.loaded) {
            outbuf[i] = 0;
            continue;
        }

        bufPtr++;
        if(bufPtr == INPUT_SIZE){
            for (int j=0; j < model.latent_size; j++){
                inBuffer[j] = in0(j + filename_length + 2);
            }
            //process block
            at::Tensor frame = torch::from_blob(inBuffer, model.latent_size);
            frame = torch::reshape(frame, {1,model.latent_size,1});
            result = model.decode(frame);
            resultData = result.data_ptr<float>();

            bufPtr = 0;
            first_block_done = true;
        }

        if (first_block_done){
            outbuf[i] = resultData[bufPtr];
        }
        else {
            outbuf[i] = 0;
        }

    }
}

} // namespace RAVE

PluginLoad(RAVEUGens) {
    // Plugin magic
    ft = inTable;
    registerUnit<RAVE::RAVE>(ft, "RAVE", false);
    registerUnit<RAVE::RAVEEncoder>(ft, "RAVEEncoder", false);
    registerUnit<RAVE::RAVEDecoder>(ft, "RAVEDecoder", false);
}
