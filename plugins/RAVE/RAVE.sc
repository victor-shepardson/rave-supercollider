RAVE : UGen {
	*new { |filename, input, usePrior, temperature|
		var file_args = Array.with(filename.size, *filename.asList.collect(_.ascii));
		var input_args = [input, usePrior, temperature];
		var inst = this.multiNew('audio', *(file_args++input_args));
		^inst
	}
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}
}

RAVEEncoder : MultiOutUGen {
	*new { |filename, input|
		var file_args = Array.with(filename.size, *filename.asList.collect(_.ascii));
		var input_args = [input];
		var inst, chan;
		#inst, chan = this.multiNew('control', *(file_args++input_args));
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
	*new { |filename, input|
		var file_args = Array.with(filename.size, *filename.asList.collect(_.ascii));
		var input_args = Array.with(input.size, *input);
		var inst = this.multiNew('audio', *(file_args++input_args));
		^inst
	}
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}
}
