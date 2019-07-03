#read_verilog sync_dual_port_ram.v
#read_verilog fifo.v
#read_verilog dsam_encoder.v
#read_verilog dsam_encoder_tb.v

#synth_design -top dsam_encoder -part xc7z020clg484-1 

verilog sync_dual_port_ram.v fifo.v dsam_encoder.v dsam_encoder_tb.v
