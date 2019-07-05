
`timescale 1ns / 1ns
  
module dsam_encoder_tb_1;

reg clk, reset;
reg  [16-1:0] data_in, correct;
wire [16-1:0] data_out;

// DUT
dsam_encoder #(
  .ADDR_WIDTH(2),
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
  $dumpvars(0,dsam_encoder_tb_1);

  data_in = 16'd25494;
  correct = 16'd0;
  #20 data_in = 16'd26034;
  correct = 16'd25494;
  #20 data_in = 16'd49697;
  correct = 16'd1572;
  #20 data_in = 16'd18638;
  correct = 16'd50181;
  #20 data_in = 16'd65016;
  correct = 16'd36043;
  #20 data_in = 16'd22891;
  correct = 16'd5801;
  #20 data_in = 16'd23632;
  correct = 16'd6894;
  #20 data_in = 16'd10049;
  correct = 16'd32575;
  #20 data_in = 16'd23478;
  correct = 16'd24242;
  #20 data_in = 16'd63761;
  correct = 16'd64752;
  #20 correct = 16'd25430;
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
