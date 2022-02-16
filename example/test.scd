ServerOptions.devices;

(
Server.default.options.inDevice_("Built-in Microph");
Server.default.options.outDevice_("Built-in Output");
s.options.sampleRate = 48000;
s.options.blockSize = 2048;
s.options.hardwareBufferSize = 2048;
s.boot
)

s.reboot

// ~bus = Bus.new(index:4, numChannels:1)
~bus = Bus.new(index:0, numChannels:1)

(
~synth = {

    var out = Limiter.ar(
        RAVE.ar(
            // SinOsc.ar(1.5,0,0.5,0.5)*Saw.ar(100)*0.9
            SoundIn.ar(0)
        , DC.ar(1))
    );
    [out, out];

}.play(outbus:~bus);

~bus.scope.window.bounds = Rect(0,1050,740,550);
)

~synth.free

// ok for some reason the unit index is 4 contrary to ~synth.trace
s.sendMsg("/u_cmd", ~synth.nodeID, 4, "/load", "/Users/victor/speech_realtime.ts")

s.queryAllNodes

~synth.trace

s.quit

s.scope

s.options.blockSize