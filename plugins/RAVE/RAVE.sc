RAVE : UGen {
	*new { |filename, input, usePrior, temperature|
		var file_args = Array.with(filename.size, *filename.asList.collect(_.ascii));
		var input_args = [input, usePrior, temperature];
		filename.isString.not.if{
			"ERROR: % first argument should be a String (the RAVE model filename)
			note that the filename does *not* support multichannel expansion"
			.format(this).postln;
		};
		^this.multiNew('audio', *(file_args++input_args));
	}
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}
}

RAVEKr : MultiOutUGen {
	// TODO: is there any way to have a dynamic number of outputs,
	// i.e. not determined until the synth is created?

	*new { |filename, latentSize ...input_args|
		var file_args = Array.with(
			filename.size, *filename.asList.collect(_.ascii));
		filename.isString.not.if{
			"ERROR: % first argument should be a String (the RAVE model filename)
			note that the filename does *not* support multichannel expansion"
			.format(this).postln;
		};
		latentSize.isInteger.not.if{
			"ERROR: % second argument should be an Integer (the RAVE model latent size)
			note that the latent size does *not* support multichannel expansion"
			.format(this).postln;
		};
		^this.multiNew('control', latentSize, *(
			file_args++[latentSize]++input_args));
	}
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}
	init { arg latentSize ...theInputs;
		inputs = theInputs;
		channels = latentSize.collect{ |i|
			OutputProxy('control', this, i)
		};
		^ channels
	}
}

RAVEPrior : RAVEKr {}
RAVEEncoder : RAVEKr {}

RAVEDecoder : UGen {
	*new { |filename, input|
		var file_args = Array.with(filename.size, *filename.asList.collect(_.ascii));
		// support multichannel expansion when passing array of latent vectors
		var flop_input = input.postln[0].postln.isSequenceableCollection.postln.if{input.flop}{input};
		var input_args = Array.with(flop_input.size, *flop_input);
		var inst = this.multiNew('audio', *(file_args++input_args));
		^inst
	}
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}
}
