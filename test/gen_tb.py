import math



header_template = '''
`timescale 1ns / 1ns
  
module dsam_encoder_tb_{test_num};

reg clk, reset;
reg  [{data_width}-1:0] data_in, correct;
wire [{data_width}-1:0] data_out;

// DUT
dsam_encoder #(
  .ADDR_WIDTH({addr_width}),
  .DATA_WIDTH({data_width}),
  .CHANNELS({channels})
) dut (
  .clk(clk),
  .reset(reset),
  .in(data_in),
  .out(data_out)
);

// clk gen
initial begin
  reset = 1;
  clk = 1;
  forever #10 clk = !clk;
end

// test cases
initial begin
  $dumpfile("test.vcd");
  $dumpvars(0,dsam_encoder_tb_{test_num});
 
'''.format(
  test_num='',
  addr_width=math.ceiling(math.log(channels,2)),
  data_width='16',
)

