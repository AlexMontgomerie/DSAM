`timescale 1ns / 1ns

module test_dsam_encoder;

reg clk, reset;
reg  [15:0] data_in, correct;
wire [15:0] data_out;

// DUT
dsam_encoder #(
  .DATA_WIDTH(16),
  .CHANNELS(4)
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
  $dumpvars(0,test_dsam_encoder);
  //data_in = 16'h0000;
  //#20 reset = 0;
  data_in = 16'h0001;
  correct = 16'h0000;
  #20 data_in = 16'h0002;
  correct     = 16'h0001;
  #20 data_in = 16'h0003;
  correct     = 16'h0003;
  #20 data_in = 16'h0004;
  correct     = 16'h0000;
  #20 data_in = 16'h0005;
  correct     = 16'h0004;
  #20 data_in = 16'h0006;
  correct     = 16'h0000;
  #20 data_in = 16'h0007;
  correct     = 16'h0004;
  #20 data_in = 16'h0008;
  correct     = 16'h0000;
  #20 
  correct     = 16'h0004;
  $finish;
end

initial begin
  $monitor($stime, " in=%h, out=%h, correct=%h",
    data_in,
    data_out,
    correct
  );
end

endmodule
