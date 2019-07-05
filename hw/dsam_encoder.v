`timescale 1ns / 1ns
module dsam_encoder #(
  parameter ADDR_WIDTH = 3,
  parameter DATA_WIDTH = 16,
  CHANNELS = 256
) (
  input  clk, reset,
  input  [DATA_WIDTH-1:0] in,
  output [DATA_WIDTH-1:0] out
);

  reg [7:0 ]cntr = 0;
  
  reg [DATA_WIDTH-2:0] corr = 0;
  reg sign_buf = 0;  

  reg [DATA_WIDTH-1:0] buffer_in;
  wire [DATA_WIDTH-1:0] buffer_out;
  reg [DATA_WIDTH-1:0] sub = 0;

  // fifo signals
  wire empty, full;
  reg read  = 0;
  reg write = 0;

  initial begin
    //sub   <= 0;
    sign_buf <= 1;
    corr  <= 16'h0000;
    write <= 1;
  end

  fifo #(
    .ADDRESS_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(DATA_WIDTH)
  ) buffer (
    .clk          (clk),
    .reset        (reset),
    .read         (read),
    .write        (write),
    .write_data   (buffer_in),
    .empty        (empty),
    .full         (full),
    .read_data    (buffer_out)  
  );  

  // sync logic
  always @ (posedge clk)
  begin
    // update fifo
    buffer_in <= in;
    // output
    sub  <= (cntr == CHANNELS) ? 
        ( (in>buffer_out) ? in - buffer_out : buffer_out - in ) : in;
    // corr <= sub[DATA_WIDTH-1] ? 
    //   ((~sub[DATA_WIDTH-2:0]) ^ corr) : 
    //   (( sub[DATA_WIDTH-2:0]) ^ corr) ;
    corr <= sub[DATA_WIDTH-2:0] ^ corr ; 
    
    sign_buf <= sub[DATA_WIDTH-1];
    
    // initial 
    if (cntr < CHANNELS-1) begin
      cntr <= cntr + 1;
    end
    else if (cntr == CHANNELS-1) begin
      cntr <= CHANNELS;
      read = 1;
    end
  end

  assign out = {sign_buf, corr};

endmodule
