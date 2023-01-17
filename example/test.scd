(
// Server.default.options.inDevice_("BlackHole 16ch");
// Server.default.options.outDevice_("BlackHole 16ch");
// Server.default.options.inDevice_("MacBook Pro Microphone");
// Server.default.options.outDevice_("MacBook Pro Speakers");
// Server.default.options.inDevice_("External Microphone");
// Server.default.options.outDevice_("External Speakers");

// Server.supernova;
Server.scsynth;

s.options.sampleRate = 48000; // the sample rate must match RAVE
s.options.blockSize = 512;//16; // a low block size within scsynth is fine,
s.options.hardwareBufferSize = 2048;//512; // but the hardware buffer size ought to match RAVE
// (to avoid dropouts). It *can* be lower for slightly less latency, at a high peak cpu cost

"OMP_NUM_THREADS".setenv("1"); // this saves a lot of CPU on arm macs, at least

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
// you can also extract RAVE models from the Neutone app: https://neutone.space/plugin/

// basic prior, timbre transfer:
(
~synth = {
    var out = Limiter.ar(
        RAVE.new(
			"/path/to/my/rave/model.ts",
            SoundIn.ar(0), // input for timbre transfer
            \prior.kr(0), // 0 for timbre transfer, 1 for generation
            \temp.kr(1) // temperature of generation
        )
    );
    out!2;

}.play(outbus:~bus);

)
// set prior to 1 to use prior instead of encoder
~synth.set(\prior, 1);
// then control the sampling temperature: 0 is deterministic, 1 is 'normal', >1 is more extreme
~synth.set(\temp, 0);


// jig to test latency:
(
~synth = {
	var in = Saw.ar(-1) * Pulse.ar(200)/2;
    var out = RAVE.new(
		"/path/to/my/rave/model.ts",
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
		"/path/to/my/rave/model.ts", 8, // filename, latent size
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
		"/path/to/my/rave/model.ts", 8, // filename, latent size
		SoundIn.ar(0),// input for latent embedding
	);
	// z[0].poll(2, \first_latent); z[1].poll(2, \second_latent);
	Limiter.ar(
		RAVEDecoder.new(
        "/path/to/my/rave/model.ts",
			[z[0]]++(1-z[1..])
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
		"/path/to/my/rave/model.ts", 8, // filename, latent size
		DC.kr(0.2), // prior temperature
	);
	// z.poll(2);
	Limiter.ar(
        RAVEDecoder.new(
			"/path/to/my/rave/model.ts",
			z))!2 // latent inputs
	// [z, z+z.collect{PinkNoise.kr*2-1/3}])) // pseudo-stereo
    ;
}.play(outbus:~bus);

)

//decoder-only + live coding latents
(
~synth = {
	var z = 36.collect{ |i| ("latent_"++i).asSymbol.kr};
	Limiter.ar(
        RAVEDecoder.new(
			"/path/to/my/rave/model.ts",
            z,
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
n = 8;
r = 48000/2048;
~set.(\latent_0, {SinOsc.kr(r/8)*1});
(n-1).do{ |i|
	var synth = {SinOsc.kr(r/(n-i).nthPrime/16)*2};
	var key = ("latent_"++(i+1)).asSymbol.postln;
	~set.(key, synth);
}
)


~synth.free

s.quit

s.stopRecording
