// PluginRAVE.cpp
// Victor Shepardson (victor.shepardson@gmail.com)

#include "SC_PlugIn.hpp"
#include "RAVE.hpp"

static InterfaceTable* ft;

namespace RAVE {

RAVE::RAVE() {
    mCalcFunc = make_calc_function<RAVE, &RAVE::next>();
    next(1);
}

void RAVE::next(int nSamples) {
    const float* input = in(0);
    const float* gain = in(1);
    float* outbuf = out(0);

    // simple gain function
    for (int i = 0; i < nSamples; ++i) {
        outbuf[i] = input[i] * gain[i];
    }
}

} // namespace RAVE

PluginLoad(RAVEUGens) {
    // Plugin magic
    ft = inTable;
    registerUnit<RAVE::RAVE>(ft, "RAVE", false);
}
