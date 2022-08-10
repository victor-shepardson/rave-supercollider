RAVE : UGen {
	*new { |control, input, usePrior, temperature|
		var inst = this.multiNew('audio', input, usePrior, temperature);
		// the RAVEControl needs to know UGen index  in the synth to send
		// the load command later
		control.ugenID = inst.synthIndex;
		^inst
	}
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}
}

RAVEEncoder : MultiOutUGen {
	*new { |control, input, usePrior, temperature|
		var inst, chan;
		#inst, chan = this.multiNew('control', input, usePrior, temperature);
		// the RAVEControl needs to know UGen index in the synth to send
		// the load command later
		control.ugenID = inst.synthIndex;
		^chan
	}
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}
	init { arg ... theInputs;
		inputs = theInputs;
		// fixed 8 latent dims for now
		// TODO: make this dynamic, matching actual number of latents in model
		channels = 8.collect{ |i|
			OutputProxy('control', this, i)
		};
		^[this, channels]
	}
}

RAVEDecoder : UGen {
	*new { |control, input|
		var inst = this.multiNew('audio', *input);
		// the RAVEControl needs to know UGen index  in the synth to send
		// the load command later
		control.ugenID = inst.synthIndex;
		^inst
	}
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}
}

RAVEControl {
	var <>server, <>modelFile, <>ugenID;

	*new { | server, modelFile|
        ^super.newCopyArgs(server, modelFile)
    }

	load { |synth|
		// call the load_model function in the UGen	
		server.sendMsg("/u_cmd", synth.nodeID, ugenID, "/load", modelFile)
	}
}