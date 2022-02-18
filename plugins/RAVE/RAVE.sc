RAVE : UGen {
	*ar { |control, input, gain|
		var inst = this.multiNew('audio', input, gain);
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