(
Server.default.options.inDevice_("Built-in Microph");
Server.default.options.outDevice_("Built-in Output");
s.options.sampleRate = 48000;
s.options.blockSize = 2048;
s.options.hardwareBufferSize = 2048;
s.boot;
~bus = Bus.new(index:0, numChannels:1);
)

// if there are code signing problems:
("xattr -d -r com.apple.quarantine"+Platform.userExtensionDir.quote++"/rave-supercollider/").runInTerminal

(
//use a RAVEControl object to manage model loading
//fill in the location of realtime RAVE model
~ctl = RAVEControl(s,
    //fill in the location of realtime RAVE model
    modelFile:"/Users/victor/speech_realtime.ts"
);
~synth = {

    var out = Limiter.ar(
        RAVE.ar(
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

// run this once the synth has been created on the server
// to load model and start the RAVE UGen
~ctl.load(~synth);

~synth.set(\prior, 1);

~synth.set(\temp, 0);

s.quit
