// RAVE.cpp
// Victor Shepardson (victor.shepardson@gmail.com)

#include "RAVE.hpp"
#include "SC_PlugIn.hpp"

static InterfaceTable* ft;

namespace RAVE {

void load_model(struct Unit* unit, struct sc_msg_iter* args){
    auto rave = (RAVE*)unit;
    const char *path = args->gets();
    rave->model.load(path);
}

RAVE::RAVE() {
    model = RAVEModel();

    inBuffer = (float*)RTAlloc(this->mWorld, INPUT_SIZE * sizeof(float));
    bufPtr = 0;
    first_block = false;

    mCalcFunc = make_calc_function<RAVE, &RAVE::next>();
    // next(INPUT_SIZE);
}

RAVE::~RAVE() {
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
            first_block = true;
        }

        if (first_block){
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
}
