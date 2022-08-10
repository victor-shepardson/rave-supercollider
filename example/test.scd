(
Server.default.options.inDevice_("Built-in Microph");
Server.default.options.outDevice_("Built-in Output");
s.options.sampleRate = 48000;
s.options.blockSize = 2048;
s.options.hardwareBufferSize = 2048;
s.waitForBoot{
	~bus = Bus.new(index:0, numChannels:1);
}
)

// if there are code signing problems:
("xattr -d -r com.apple.quarantine"+Platform.userExtensionDir.quote++"/rave-supercollider/").runInTerminal


// some checkpoints:
// https://play.forum.ircam.fr/rave-vst-api/get_available_models
// https://play.forum.ircam.fr/rave-vst-api/get_model?model_name=

(
//use a RAVEControl object to manage model loading
//fill in the location of realtime RAVE model
~ctl = RAVEControl(s,
    //fill in the location of realtime RAVE model
    modelFile:"/Users/victor/rave-models/speech/speech_realtime.ts"
);
~synth = {

    var out = Limiter.ar(
        RAVE.new(
            ~ctl, //a RAVEControl
            SoundIn.ar(0),// input for timbre transfer
            \prior.kr(0), // 0 for timbre transfer, 1 for generation
            \temp.kr(1) // temperature of generation
        )
    );
    out!2;

}.play(outbus:~bus);

// ~bus.scope.window.bounds = Rect(0,1050,740,550);
)



(
//use a RAVEControl object to manage model loading
//fill in the location of realtime RAVE model
~ctl = RAVEControl(s,
    //fill in the location of realtime RAVE model
    modelFile:"/Users/victor/rave-models/speech/speech_realtime.ts"
);
~synth = {

	var z = RAVEEncoder.new(
		~ctl, //a RAVEControl
		SoundIn.ar(0),// input for latent embedding
		\prior.kr(0), // 0 for embedding, 1 for generation
		\temp.kr(1) // temperature of generation
	);
	z.poll(2);
	DC.ar(0)!2;
	/*Limiter.ar(
        RAVEDecoder.new(
            ~ctl, //a RAVEControl
            z,// latent inputs
        )
    )!2;*/
}.play(outbus:~bus);

)

(
//use a RAVEControl object to manage model loading
//fill in the location of realtime RAVE model
~ctl = RAVEControl(s,
    //fill in the location of realtime RAVE model
    modelFile:"/Users/victor/rave-models/speech/speech_realtime.ts"
);
~synth = {

	// var z = SinOsc.kr(8.collect{ |i| 1.5**(0-i) * 1});
	var z = 8.collect{ |i| ("latent_"++i).asSymbol.kr};
	// var z = [0,0,0,0,0,0,0,SinOsc.kr(0.5)*3];
	Limiter.ar(
        RAVEDecoder.new(
            ~ctl, //a RAVEControl
            z,// latent inputs
        )
    )!2;
}.play(outbus:~bus);

)


(
r = 48000/2048;
b = 0;
m = 0.1;
~synth.set(\latent_7, Ndef(\l7, {SinOsc.kr(r/3)*2}));
~synth.set(\latent_6, Ndef(\l6, {SinOsc.kr(r/4)*2}));
~synth.set(\latent_5, Ndef(\l5, {SinOsc.kr(r/5)*2}));
~synth.set(\latent_4, Ndef(\l4, {SinOsc.kr(r/7)*2}));
~synth.set(\latent_3, Ndef(\l3, {SinOsc.kr(r/6)*2}));
~synth.set(\latent_2, Ndef(\l2, {SinOsc.kr(r/5)*2}));
~synth.set(\latent_1, Ndef(\l1, {SinOsc.kr(r/4)*2}));
~synth.set(\latent_0, Ndef(\l0, {SinOsc.kr(r/3)*2}));
)



// run this once the synth has been created on the server
// to load model and start the RAVE UGen
~ctl.load(~synth);

~synth.set(\prior, 1);

~synth.set(\temp, 0);

s.quit
