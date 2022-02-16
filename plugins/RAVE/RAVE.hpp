// PluginRAVE.hpp
// Victor Shepardson (victor.shepardson@gmail.com)

#pragma once

#include "SC_PlugIn.hpp"

namespace RAVE {

class RAVE : public SCUnit {
public:
    RAVE();

    // Destructor
    // ~RAVE();

private:
    // Calc function
    void next(int nSamples);

    // Member variables
};

} // namespace RAVE
