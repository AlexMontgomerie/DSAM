# DSAM
A coding scheme for reducing switching activity in convolutional neural networks.

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
