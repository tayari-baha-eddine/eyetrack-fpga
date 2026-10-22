# Technical Limitations and Clarifications

## Dataset and evaluation protocol
The original Swirski dataset contains 3,760 frames, but the pipeline uses the annotated subset. Of the 600 annotated images, 24 were excluded because their pupil centers fell outside the fixed 400 × 400 ROI, leaving 576 samples. The split assigns 240 images from participant p1 to training, 60 to validation, and 276 from participant p2 to testing. This participant-disjoint protocol evaluates generalization to an unseen participant, subject to the correctness of the dataset metadata and split implementation.

## Quantization
On the 60-image validation set, mean localization error increases from 4.06 px with floating-point inference to 5.17 px with integer inference. The mean and maximum differences between floating-point and quantized predictions are reported separately as 3.22 px and 5.88 px, respectively. Quantization of intermediate activations is a potential source of this discrepancy, but the contribution of each layer has not yet been isolated experimentally.

## Accumulator width
INT32 accumulators are used to provide additional numerical headroom and simplify implementation. The first-layer raw-product bound under the stated input and weight assumptions is approximately 8.29 million, close to the positive limit of a signed 24-bit integer. A reduction to 24 bits would require a complete range analysis including quantized biases, signed arithmetic, intermediate values, and overflow behavior.

## Synthesis limitations
GHDL synthesis was used to assess whether the VHDL design can be synthesized into a generic hardware representation. Memory inference in this representation does not establish how a specific FPGA toolchain will map weights to block RAM, distributed RAM, or LUT logic. No target-specific synthesis, place-and-route, physical timing, or power measurements have been performed.

## Latency
The reported latency is 4,244 clock cycles according to the simulation testbench. At a hypothetical 100 MHz clock frequency, this corresponds to 42.44 µs. This is a theoretical conversion, not a measured FPGA latency; the cycle count depends on the verified start/done measurement protocol.
