(
// Server.default.options.inDevice_("BlackHole 16ch");
// Server.default.options.outDevice_("BlackHole 16ch");
Server.default.options.inDevice_("MacBook Pro Microphone");
Server.default.options.outDevice_("MacBook Pro Speakers");

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

// if macOS complains about unknown binaries:
("xattr -d -r com.apple.quarantine"+Platform.userExtensionDir.quote++"/RAVE/").runInTerminal


// some checkpoints:
// https://play.forum.ircam.fr/rave-vst-api/get_available_models
// https://play.forum.ircam.fr/rave-vst-api/get_model?model_name=

// old API:
(
~synth = {

    var out = Limiter.ar(
        RAVE.new(
			// "/Users/victor/Downloads/epoch=176.ckpt",
			// "/Users/victor/Downloads/rave_vctk-low-latency-hilr-lokld-vae-0215.ts",
			"/Users/victor/rave-models/speech/speech_realtime.ts",
			// "/Users/victor/Library/Application\ Support/Qosmo/Neutone/models/83ecd7d0f50c23bcb0f2d3ec12872dd9",
			// "/Users/victor/interactive-voice/RAVE/rave_cropped.ts",
            SoundIn.ar(0), // input for timbre transfer
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
		// "/Users/victor/Downloads/rave_vctk-low-latency-hilr-lokld-vae-0215.ts",
		"/Users/victor/interactive-voice/RAVE/rave_cropped.ts",
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
		// "/Users/victor/interactive-voice/RAVE/rave_cropped.ts", 8,
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
		// "/Users/victor/rave-models/speech/speech_realtime.ts", 8, // filename, latent size
		// "/Users/victor/Library/Application\ Support/Qosmo/Neutone/models/83ecd7d0f50c23bcb0f2d3ec12872dd9", 8,
		"/Users/victor/interactive-voice/RAVE/rave_cropped.ts", 36,
		SoundIn.ar(0),// input for latent embedding
		// SoundIn.ar(0)!2,// input for latent embedding
	);
	// z[0].poll(2, \A); z[1].poll(2, \B);
	// DC.ar(0)!2;
	Limiter.ar(
		RAVEDecoder.new(
			// "/Users/victor/rave-models/animal2rt_3M_sc.ts",
			// "/Users/victor/rave-models/speech/speech_realtime.ts",
			// "/Users/victor/Library/Application\ Support/Qosmo/Neutone/models/83ecd7d0f50c23bcb0f2d3ec12872dd9",
			"/Users/victor/interactive-voice/RAVE/rave_cropped.ts",
			// z
			[z[0]]++(0-z[1..]) // latent inputs
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
		DC.kr(0.2), // prior temperature
	);
	z.poll(2);
	// DC.ar(0)!2;
	Limiter.ar(
        RAVEDecoder.new(
			// "/Users/victor/rave-models/speech/speech_realtime.ts",
			// "/Users/victor/rave-models/cleansing/2M46_cleansing_small_sc.ts",
			"/Users/victor/interactive-voice/RAVE/rave_cropped.ts",
			z*2))!2 // latent inputs
	// [z, z+z.collect{PinkNoise.kr*2-1/3}])) // pseudo-stereo
    ;
}.play(outbus:~bus);

)

//decoder only
(
~synth = {

	// var z = SinOsc.kr(8.collect{ |i| 1.5**(0-i) * 1});
	var z = 36.collect{ |i| ("latent_"++i).asSymbol.kr};
	// var z = [0,0,0,0,0,0,0,SinOsc.kr(0.5)*3];
	Limiter.ar(
        RAVEDecoder.new(
			// "/Users/victor/rave-models/speech/speech_realtime.ts",
			// "/Users/victor/rave-models/speech/speech_realtime💋.ts", //test unicode
			// "/Users/victor/rave-models/wheel/wheel.ts",
			// "/Users/victor/rave-models/animal2rt_3M_sc.ts",
			// "/Users/victor/rave-models/blonk/blonk_vae_only_sc.ts",
			// "/Users/victor/rave-models/blonk/blonk_2300865_sc.ts",
			"/Users/victor/interactive-voice/RAVE/rave_cropped.ts",

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


(
r = 48000/2048;
36.do{ |i|
	var synth = {SinOsc.kr(r/i.nthPrime/(i>0).if{64}{2})*2.5};
	var key = ("latent_"++i).asSymbol.postln;
	~set.(key, synth);
}
)

(
n = 36;
r = 48000/2048;
~set.(\latent_0, {SinOsc.kr(r/4)*3});
(n-1).do{ |i|
	var synth = {SinOsc.kr(r/(n-i).nthPrime/8)*2.5};
	var key = ("latent_"++(i+1)).asSymbol.postln;
	~set.(key, synth);
}
)


36.collect(_.nthPrime.asFloat).product * 64 / r / 60 / 60 / 24 / 365

~synth.free

s.quit
