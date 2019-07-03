# DSAM
A coding scheme for reducing switching activity in convolutional neural networks.

## Simulation
Coding schemes are simulated in python. To run simulation, use the following commands:

```
python3 run_encoding.py -m <model path> -d <data path> -w <weight path> -n <number of images> -o <offset file>
    --base = baseline
    --binv = Bus Invert coding scheme
    --dsam = DSAM coding scheme
    --csam = CSAM coding scheme
    --apbm = APBM coding scheme

```

Example usage:

```
python3 sim/run_encoding.py -m model/lenet.prototxt -d data/mnist/ -w weight/lenet.caffemodel -n 10 -o coef/lenet.json --base
```



## Run Testbenches

```
iverilog -o test dsam_encoder.v dsam_encoder_tb.v fifo.v sync_dual_port_ram.v
vvp test 
``` 

And to view the waveform

```
gtkwave test.vcd
```

### TODO
 - testbench gen
 - verify encoding scheme
 - Add reset signal
