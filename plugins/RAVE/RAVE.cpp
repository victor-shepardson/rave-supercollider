// RAVE.cpp
// Victor Shepardson (victor.shepardson@gmail.com)

#include "RAVE.hpp"
#include "SC_PlugIn.hpp"

static InterfaceTable* ft;

namespace RAVE {

void load_model(struct Unit* unit, struct sc_msg_iter* args){
    auto rave = (RAVEBase*)unit;
    const char *path = args->gets();
    rave->model.load(path);
}

RAVEBase::RAVEBase() {
    model = RAVEModel();

    bufPtr = 0;
    first_block_done = false;
}

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
}

RAVEBase::~RAVEBase() {
    RTFree(this->mWorld, inBuffer);
}

void RAVE::next(int nSamples) {
    const float* input = in(0);
    const float use_prior = in0(1);
    const float temperature = in0(2);

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
            at::Tensor frame = torch::from_blob(inBuffer, INPUT_SIZE);
            if(use_prior){
                result = model.sample_from_prior(temperature);
            } else {
                frame = torch::reshape(frame, {1,1,INPUT_SIZE});
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
    const float* input = in(0);
    const float use_prior = in0(1);
    const float temperature = in0(2);

    // should write zeros here but don't know how many outputs before model is loaded
    // leaving this since we want to change how model loading works anyway...
    if (!model.loaded) return;

    for (int i = 0; i < fullBufferSize(); ++i) {

        inBuffer[bufPtr] = input[i];
        bufPtr++;
        if(bufPtr == INPUT_SIZE){
            //process block
            at::Tensor frame = torch::from_blob(inBuffer, INPUT_SIZE);
            if(use_prior){
                result = model.sample_from_prior(temperature);
            } else {
                frame = torch::reshape(frame, {1,1,INPUT_SIZE});
                result = model.encode(frame);
            }
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
                inBuffer[j] = in0(j);
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
    DefineUnitCmd("RAVE", "/load", RAVE::load_model);
    registerUnit<RAVE::RAVEEncoder>(ft, "RAVEEncoder", false);
    DefineUnitCmd("RAVEEncoder", "/load", RAVE::load_model);
    registerUnit<RAVE::RAVEDecoder>(ft, "RAVEDecoder", false);
    DefineUnitCmd("RAVEDecoder", "/load", RAVE::load_model);
}
