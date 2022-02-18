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
~ctl = RAVEControl(s, modelFile:"/Users/victor/speech_realtime.ts");
~synth = {

    var out = Limiter.ar(
        RAVE.ar(
            ~ctl, // first argument is a RAVEControl
            SoundIn.ar(0), //encoder input
            DC.ar(1)) // temperature
    );
    out!2;

}.play(outbus:~bus);

// ~bus.scope.window.bounds = Rect(0,1050,740,550);
)

// run this once the synth has been created on the server
// load model and start the RAVE UGen
~ctl.load(~synth);

s.quit
