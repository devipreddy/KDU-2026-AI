class PCMChunkProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetSampleRate = options.processorOptions.targetSampleRate || 24000;
    this.chunkFrames = options.processorOptions.chunkFrames || 6000;
    this.pending = [];
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) {
      return true;
    }
    const channel = input[0];
    const resampled = this.resample(channel, sampleRate, this.targetSampleRate);
    for (let i = 0; i < resampled.length; i += 1) {
      this.pending.push(resampled[i]);
    }
    while (this.pending.length >= this.chunkFrames) {
      const slice = this.pending.splice(0, this.chunkFrames);
      const pcm = new Int16Array(slice.length);
      let energy = 0;
      for (let index = 0; index < slice.length; index += 1) {
        const sample = Math.max(-1, Math.min(1, slice[index]));
        pcm[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
        energy += sample * sample;
      }
      const rms = Math.sqrt(energy / slice.length);
      this.port.postMessage({ pcm: pcm.buffer, rms }, [pcm.buffer]);
    }
    return true;
  }

  resample(buffer, fromRate, toRate) {
    if (fromRate === toRate) {
      return Float32Array.from(buffer);
    }
    const ratio = fromRate / toRate;
    const newLength = Math.round(buffer.length / ratio);
    const output = new Float32Array(newLength);
    for (let i = 0; i < newLength; i += 1) {
      const start = Math.floor(i * ratio);
      const end = Math.min(buffer.length, Math.floor((i + 1) * ratio));
      let total = 0;
      let count = 0;
      for (let j = start; j < end; j += 1) {
        total += buffer[j];
        count += 1;
      }
      output[i] = count ? total / count : 0;
    }
    return output;
  }
}

registerProcessor("pcm-chunk-processor", PCMChunkProcessor);
