(
Server.default.options.inDevice_("BlackHole 16ch");
Server.default.options.outDevice_("BlackHole 16ch");
// Server.default.options.inDevice_("Built-in Microph");
// Server.default.options.outDevice_("Built-in Output");

// Server.supernova;
Server.scsynth;

s.options.sampleRate = 48000; // the sample rate must match RAVE
s.options.blockSize = 512;//16; // a low block size within scsynth is fine,
s.options.hardwareBufferSize = 2048; // but the hardware buffer size needs to match RAVE
// (to avoid dropouts)

s.waitForBoot{
	~bus = Bus.new(index:0, numChannels:2);
}
)

s.quit

// if there are code signing problems:
("xattr -d -r com.apple.quarantine"+Platform.userExtensionDir.quote++"/rave-supercollider/").runInTerminal


// some checkpoints:
// https://play.forum.ircam.fr/rave-vst-api/get_available_models
// https://play.forum.ircam.fr/rave-vst-api/get_model?model_name=

// old API:
(
~synth = {

    var out = Limiter.ar(
        RAVE.new(
			"/Users/victor/Downloads/rave_vctk-low-latency-hilr-lokld-vae-0215.ts",
			// "/Users/victor/rave-models/speech/speech_realtime.ts",
			// "/Users/victor/Library/Application\ Support/Qosmo/Neutone/models/83ecd7d0f50c23bcb0f2d3ec12872dd9", // does not work -- neutone replaces `forward` i think
            SoundIn.ar(0), // input for timbr==e transfer
            \prior.kr(0), // 0 for timbre transfer, 1 for generation
            \temp.kr(1) // temperature of generation
        )
    );
    out!2;

}.play(outbus:~bus);

// ~bus.scope.window.bounds = Rect(0,1050,740,550);
)
~synth.set(\prior, 1);
~synth.set(\temp, 0);


// test latency of 'no-latency' model
// ~100ms?
(
~synth = {
	// var in = Blip.ar(1, 150)*Pulse.ar(200)/2;
	var in = Saw.ar(-1) * Pulse.ar(200)/2;
    var out = RAVE.new(
		"/Users/victor/Downloads/rave_vctk-low-latency-hilr-lokld-vae-0215.ts",
		in, // input for timbre transfer
		\prior.kr(0), // 0 for timbre transfer, 1 for generation
		\temp.kr(1) // temperature of generation
	);
	Limiter.ar([in, out], 0.2);
}.play(outbus:~bus);
)



// Encoder only
// RAVEEncoder needs the latent size as an argument, since it needs to create that many
// control rate outputs.
(
~synth = {
	var z = RAVEEncoder.new(
		"/Users/victor/rave-models/speech/speech_realtime.ts", 8, // filename, latent size
		SoundIn.ar(0),// input for latent embedding
	);
	z.poll(2,);
	DC.ar(0)!2
}.play(outbus:~bus);

)


// separate encoder/decoder
// RAVEEncoder needs the latent size as an argument, since it needs to create that many
// control rate outputs.
// RAVEDecoder can handle a mismatched input size
// (extra inputs are ignored, extra latent dimensions are zeroed)
(
~synth = {
	var z = RAVEEncoder.new(
		// "/Users/victor/rave-models/animal2rt_3M_sc.ts", 4,
		"/Users/victor/rave-models/speech/speech_realtime.ts", 8, // filename, latent size
		// "/Users/victor/Library/Application\ Support/Qosmo/Neutone/models/83ecd7d0f50c23bcb0f2d3ec12872dd9", 8,
		SoundIn.ar(0),// input for latent embedding
		// SoundIn.ar(0)!2,// input for latent embedding
	);
	// z[0].poll(2, \A); z[1].poll(2, \B);
	// DC.ar(0)!2;
	Limiter.ar(
		RAVEDecoder.new(
			// "/Users/victor/rave-models/animal2rt_3M_sc.ts",
			"/Users/victor/rave-models/speech/speech_realtime.ts",
			// "/Users/victor/Library/Application\ Support/Qosmo/Neutone/models/83ecd7d0f50c23bcb0f2d3ec12872dd9",
			z
			// 0-z.shift(1) // latent inputs
		)!2
	);
}.play(outbus:~bus);

)


// separate prior/decoder
// RAVEPrior needs the latent size as an argument, since it needs to create that many
// control rate outputs.
// RAVEDecoder can handle a mismatched input size
// (extra inputs are ignored, extra latent dimensions are zeroed)
(
~synth = {
	var z = RAVEPrior.new(
		"/Users/victor/rave-models/speech/speech_realtime.ts", 8, // filename, latent size
		DC.kr(0.7), // prior temperature
	);
	z.poll(2);
	// DC.ar(0)!2;
	Limiter.ar(
        RAVEDecoder.new(
			// "/Users/victor/rave-models/speech/speech_realtime.ts",
			"/Users/victor/rave-models/cleansing/2M46_cleansing_small_sc.ts",
			z))!2 // latent inputs
	// [z, z+z.collect{PinkNoise.kr*2-1/3}])) // pseudo-stereo
    ;
}.play(outbus:~bus);

)

//decoder only
(
~synth = {

	// var z = SinOsc.kr(8.collect{ |i| 1.5**(0-i) * 1});
	var z = 8.collect{ |i| ("latent_"++i).asSymbol.kr};
	// var z = [0,0,0,0,0,0,0,SinOsc.kr(0.5)*3];
	Limiter.ar(
        RAVEDecoder.new(
			"/Users/victor/rave-models/speech/speech_realtime.ts",
			// "/Users/victor/rave-models/speech/speech_realtime💋.ts", //test unicode
			// "/Users/victor/rave-models/wheel/wheel.ts",
			// "/Users/victor/rave-models/animal2rt_3M_sc.ts",
			// "/Users/victor/rave-models/blonk/blonk_vae_only_sc.ts",
			// "/Users/victor/rave-models/blonk/blonk_2300865_sc.ts",
            z,// latent inputs
        )
    )!2;
}.play(outbus:~bus);

~set = { |key, fn| ~synth.set(key, Ndef(key, fn))};
)


(
r = 48000/2048;
b = 0;
m = 0.1;
~set.(\latent_0, {SinOsc.kr(r/4)*2});
~set.(\latent_1, {SinOsc.kr(r/3)*2});
~set.(\latent_2, {SinOsc.kr(r/7)*2});
~set.(\latent_3, {SinOsc.kr(r/7)*2});
~set.(\latent_4, {SinOsc.kr(r/11)*2});
~set.(\latent_5, {SinOsc.kr(r/13)*2});
~set.(\latent_6, {SinOsc.kr(r/17)*2});
~set.(\latent_7, {SinOsc.kr(r/19)*2});
)

~synth.free

s.quit
