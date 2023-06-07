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
    load_thread = std::make_unique<std::thread>(&RAVEModel::load, model, path);
}

RAVEBase::~RAVEBase() {
    if (load_thread && load_thread->joinable()) {
        compute_thread->join();
    }
    if (compute_thread && compute_thread->joinable()) {
        compute_thread->join();
    }
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

AsyncRAVEBase::AsyncRAVEBase() : RAVEBase() {
    // TODO: expose processing latency to user
    //defaults to block_size - 1
    m_processing_latency = -1;
    // count of model samples elapsed
    m_internal_samples = 0;
}

AsyncRAVEBase::~AsyncRAVEBase() {
    //TODO: sDtor inheritance?
    RTFree(this->mWorld, modelInBuffer);
    RTFree(this->mWorld, modelOutBuffer);
}

void AsyncRAVEBase::make_buffers(){
    // this gets called when after model loading is done
    // and before processing starts

    // TODO: expose delay to user
    delay = (model->block_size + m_processing_latency)/model->sr;
    auto buf_size = model->block_size * sizeof(float);
    inBuffer = (float*)RTAlloc(mWorld, buf_size);
    modelInBuffer = (float*)RTAlloc(mWorld, buf_size);
    outBuffer = (float*)RTAlloc(mWorld, buf_size);
    modelOutBuffer = (float*)RTAlloc(mWorld, buf_size);

    auto sr = mWorld->mFullRate.mSampleRate;
       
    if (audio_in){
        res_in = Resampler(sr, model->sr, 3);
        inIdx = 0;
        delay += res_in.delay;
    }
    else {
        // dummy resampler which just counts model blocks
        res_in = Resampler(sr, model->sr, 0);
        // start processing immediately
        inIdx = model->block_size - 1;
    }

    if (audio_out){
        res_out = Resampler(model->sr, sr, 3);
        delay += res_out.delay;
    } 
    else {
        res_out = Resampler(model->sr, sr, 0);

        if (ugen_outputs != model->latent_size){
            std::cout << "WARNING: UGen outputs (" << ugen_outputs 
            << ") do not match number of latent dimensions in model (" 
            << model->latent_size << ")" << std::endl;
        }
    }

    if (m_processing_latency < 0){
        m_processing_latency = model->block_size - 1;
    }
}

void AsyncRAVEBase::write(float x){
    // if there is no audio in,
    // still use a dummy resampler to count model blocks

    // write to the resampler
    // std::cout << "write to res_in" << std::endl;
    res_in.write(x);

    // while there are new model samples, write them into buffer
    while(res_in.pending()){
        // std::cout << "read from res_in" << std::endl;
        auto x = res_in.read();
        if(audio_in) inBuffer[inIdx] = x;
        inIdx += 1;
        m_internal_samples += 1;
        if (inIdx == model->block_size){
            dispatch();
            inIdx = 0;
        }
    }
}

float AsyncRAVEBase::read(){
    float x = 0;
    // until an output sample is ready, process model samples
    while (!res_out.pending()){
        // write zeros until first buffer is full,
        // plus processing time has elapsed
        if (m_internal_samples >= model->block_size + m_processing_latency){
            if (outIdx % model->block_size == 0){
                join();
                outIdx = 0;
            }
            if(audio_out) x = outBuffer[outIdx];
            outIdx += 1;
        }    
        // std::cout << "write to res_out" << std::endl;
        res_out.write(x);
    }
    // std::cout << "read from res_out" << std::endl;
    return res_out.read();
}

// same for AsyncRAVE and AsyncRAVEDecoder
void AsyncRAVEBase::join(){
    // join model thread
    // std::cout << "join" << std::endl;
    if (!compute_thread) 
        std::cout << "ERROR: no compute thread" << std::endl;
    if (!compute_thread->joinable()) 
        std::cout << "ERROR: compute thread not joinable" << std::endl;

    compute_thread->join();

    // swap buffers
    auto temp = modelOutBuffer;
    modelOutBuffer = outBuffer;
    outBuffer = temp;
}

AsyncRAVE::AsyncRAVE() : AsyncRAVEBase(){
    ugen_outputs = 1;
    ugen_inputs = 1;
    if (!load_thread) make_buffers();
    mCalcFunc = make_calc_function<AsyncRAVE, &AsyncRAVE::next>();
}

void AsyncRAVE::next(int nSamples) {
    // std::cout << "next" << std::endl;
    // return; //DEBUG

    const float* input = in(filename_length+1);
    // const float use_prior = in0(filename_length+2);
    // const float temperature = in0(filename_length+3);

    float* output = out(0);

    if (!model->loaded){
        for (int i=0; i<nSamples; ++i) {
            write_zeros_ar(i);
        }
        return;
    }
    else if (load_thread && load_thread->joinable()){
        load_thread->join();
        make_buffers();
    }

    for (int i=0; i<nSamples; ++i){
        output[i] = step(input[i]);
    }
}

void AsyncRAVE::dispatch(){
    const float use_prior = in0(filename_length+2);
    const float temperature = in0(filename_length+3);

    // std::cout << "dispatch" << std::endl;

    if (compute_thread && compute_thread->joinable()) 
        std::cout << "ERROR: trying to start compute_thread before previous one is finished" << std::endl;

    // swap buffers
    auto temp = modelInBuffer;
    modelInBuffer = inBuffer;
    inBuffer = temp;

    if(use_prior && model->prior_temp_size>0){
        compute_thread = std::make_unique<std::thread>(
            &RAVEModel::prior_decode, model, temperature, modelOutBuffer);
    } else {
        compute_thread = std::make_unique<std::thread>(
            &RAVEModel::encode_decode, model, modelInBuffer, modelOutBuffer);
    }          
}

AsyncRAVEEncoder::AsyncRAVEEncoder() : AsyncRAVEBase(){
    ugen_inputs = 1;
    ugen_outputs = in0(filename_length+1);
    if (!load_thread) make_buffers();
    mCalcFunc = make_calc_function<AsyncRAVEEncoder, &AsyncRAVEEncoder::next>();
}

void AsyncRAVEEncoder::next(int nSamples) {
    const float* input = in(filename_length+2);

    if (!model->loaded){
        write_zeros_kr();
        return;
    }
    else if (load_thread && load_thread->joinable()){
        load_thread->join();
        make_buffers();
    }

    for (int i=0; i<fullBufferSize(); ++i){
        step(input[i]);
    }
}

void AsyncRAVEEncoder::dispatch(){

    if (compute_thread && compute_thread->joinable()) 
        std::cout 
        << "ERROR: trying to start compute thread before previous is finished" 
        << std::endl;

    // swap buffers
    auto temp = modelInBuffer;
    modelInBuffer = inBuffer;
    inBuffer = temp;

    compute_thread = std::make_unique<std::thread>(
        &RAVEModel::encode, model, modelInBuffer, modelOutBuffer);
}

void AsyncRAVEEncoder::join(){
    // join model thread
    // std::cout << "join" << std::endl;
    if (!compute_thread) 
        std::cout << "ERROR: no compute_thread" << std::endl;
    if (!compute_thread->joinable()) 
        std::cout << "ERROR: compute thread not joinable" << std::endl;

    compute_thread->join();

    // immediately write control-rate outputs
    // SC preserves these between frames (right?)
    // if that isn't true, will need to swap the output buffers and do this in `next` instead
    for (int j=0; j < std::min(model->latent_size, this->ugen_outputs); ++j){
        out0(j) = modelOutBuffer[j];
    }
}

AsyncRAVEDecoder::AsyncRAVEDecoder() : AsyncRAVEBase(){
    ugen_inputs = in0(filename_length+1);
    ugen_outputs = 1;
    if (!load_thread) make_buffers();
    mCalcFunc = make_calc_function<AsyncRAVEDecoder, &AsyncRAVEDecoder::next>();
}

void AsyncRAVEDecoder::next(int nSamples) {
    float* output = out(0);

    if (!model->loaded){
        for (int i=0; i<nSamples; ++i) {
            write_zeros_ar(i);
        }
        return;
    }
    else if (load_thread && load_thread->joinable()){
        load_thread->join();
        make_buffers();
        std::cout << 
            "RAVEDecoder latent size: " << model->latent_size << 
            "; found " << ugen_inputs << " inputs" << std::endl;
    }

    for (int i=0; i<nSamples; ++i){
        output[i] = step(0);
    }
}

void AsyncRAVEDecoder::dispatch(){
    const float use_prior = in0(filename_length+2);
    const float temperature = in0(filename_length+3);

    // std::cout << "dispatch" << std::endl;

    if (compute_thread && compute_thread->joinable()) 
        std::cout << "ERROR: trying to start compute_thread before previous one is finished" << std::endl;

    // assumption: n_latent <= block_size

    // read control rate latents
    int first_input = filename_length + 2;
    for (int j=0; j < model->latent_size; ++j){
        if (j<ugen_inputs){
            inBuffer[j] = in0(j + first_input);
        } else{
            inBuffer[j] = 0;
        }
    }

    compute_thread = std::make_unique<std::thread>(
        &RAVEModel::decode, model, inBuffer, modelOutBuffer);    
}


AsyncRAVEPrior::AsyncRAVEPrior() : AsyncRAVEBase(){
    ugen_inputs = 1;
    ugen_outputs = in0(filename_length+1);
    if (!load_thread) make_buffers();
    mCalcFunc = make_calc_function<AsyncRAVEPrior, &AsyncRAVEPrior::next>();
}

void AsyncRAVEPrior::next(int nSamples) {
    if (!model->loaded){
        write_zeros_kr();
        return;
    }
    else if (load_thread && load_thread->joinable()){
        load_thread->join();
        make_buffers();
    }

    for (int i=0; i<fullBufferSize(); ++i){
        step(0);
    }
}

void AsyncRAVEPrior::dispatch(){
    const float temperature = in0(filename_length+2);

    if (compute_thread && compute_thread->joinable()) 
        std::cout 
        << "ERROR: trying to start compute thread before previous is finished" 
        << std::endl;

    compute_thread = std::make_unique<std::thread>(
        &RAVEModel::prior, model, temperature, modelOutBuffer);
}

void AsyncRAVEPrior::join(){
    // join model thread
    // std::cout << "join" << std::endl;
    if (!compute_thread) 
        std::cout << "ERROR: no compute_thread" << std::endl;
    if (!compute_thread->joinable()) 
        std::cout << "ERROR: compute thread not joinable" << std::endl;

    compute_thread->join();

    // immediately write control-rate outputs
    // SC preserves these between frames (right?)
    // if that isn't true, will need to swap the output buffers and do this in `next` instead
    for (int j=0; j < std::min(model->latent_size, ugen_outputs); ++j){
        out0(j) = modelOutBuffer[j];
    }
}


/////////////////////////

void RAVE::make_buffers(){
    inBuffer = (float*)RTAlloc(this->mWorld, model->block_size * sizeof(float));
    outBuffer = (float*)RTAlloc(this->mWorld, model->block_size * sizeof(float));
}
RAVE::RAVE() : RAVEBase(){
    this->ugen_outputs = 1;
    if (!load_thread) make_buffers();
    mCalcFunc = make_calc_function<RAVE, &RAVE::next>();
}

void RAVEPrior::make_buffers(){
    inBuffer = (float*)RTAlloc(this->mWorld, model->latent_size * sizeof(float));
    outBuffer = (float*)RTAlloc(this->mWorld, model->latent_size * sizeof(float));
    if (ugen_outputs != model->latent_size){
        std::cout << "WARNING: UGen outputs (" << ugen_outputs << ") do not match number of latent dimensions in model (" << model->latent_size << ")" << std::endl;
    }
}
RAVEPrior::RAVEPrior() : RAVEBase(){
    this->ugen_outputs = in0(filename_length+1);
    if (!load_thread) make_buffers();
    // currently unused but is freed in superclass destructor
    mCalcFunc = make_calc_function<RAVEPrior, &RAVEPrior::next>();
}

void RAVEEncoder::make_buffers(){
    inBuffer = (float*)RTAlloc(this->mWorld, model->block_size * sizeof(float));
    outBuffer = (float*)RTAlloc(this->mWorld, model->latent_size * sizeof(float));

    if (ugen_outputs != model->latent_size){
        std::cout << "WARNING: UGen outputs (" << ugen_outputs << ") do not match number of latent dimensions in model (" << model->latent_size << ")" << std::endl;
    }
}

RAVEEncoder::RAVEEncoder() : RAVEBase(){
    this->ugen_outputs = in0(filename_length+1);
    if (!load_thread) make_buffers();
    mCalcFunc = make_calc_function<RAVEEncoder, &RAVEEncoder::next>();
}

void RAVEDecoder::make_buffers(){
    inBuffer = (float*)RTAlloc(this->mWorld, model->latent_size * sizeof(float));
    outBuffer = (float*)RTAlloc(this->mWorld, model->block_size * sizeof(float));
    std::cout << 
        "RAVEDecoder latent size: " << model->latent_size << 
        "; found " << ugen_inputs << " inputs" << std::endl;
}

RAVEDecoder::RAVEDecoder() : RAVEBase(){
    // number of inputs provided in synthdef
    // filename len, *chars, inputs len, *inputs
    this->ugen_inputs = in0(filename_length+1);
    this->ugen_outputs = 1;
    if (!load_thread) make_buffers();
    mCalcFunc = make_calc_function<RAVEDecoder, &RAVEDecoder::next>();
}

void RAVE::next(int nSamples) {
    const float* input = in(filename_length+1);
    const float use_prior = in0(filename_length+2);
    const float temperature = in0(filename_length+3);

    float* output = out(0);

    if (!model->loaded){
        for (int i = 0; i < nSamples; ++i) {
            write_zeros_ar(i);
        }
        return;
    }
    else if (load_thread && load_thread->joinable()){
        load_thread->join();
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

    if (!model->loaded){
        write_zeros_kr();
        return;
    }
    else if (load_thread && load_thread->joinable()){
        load_thread->join();
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
    else if (load_thread && load_thread->joinable()){
        load_thread->join();
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
    else if (load_thread && load_thread->joinable()){
        load_thread->join();
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
    registerUnit<RAVE::AsyncRAVE>(ft, "RAVE", false);
    registerUnit<RAVE::AsyncRAVEEncoder>(ft, "RAVEEncoder", false);
    registerUnit<RAVE::AsyncRAVEDecoder>(ft, "RAVEDecoder", false);
    registerUnit<RAVE::AsyncRAVEPrior>(ft, "RAVEPrior", false);

    // registerUnit<RAVE::RAVE>(ft, "RAVE", false);
    // registerUnit<RAVE::RAVEPrior>(ft, "RAVEPrior", false);
    // registerUnit<RAVE::RAVEEncoder>(ft, "RAVEEncoder", false);
    // registerUnit<RAVE::RAVEDecoder>(ft, "RAVEDecoder", false);
}
