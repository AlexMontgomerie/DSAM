module dsam_encoder #(
  parameter DATA_WIDTH = 16,
  CHANNELS = 256
) (
  input  clk, reset,
  input  [DATA_WIDTH-1:0] in,
  output [DATA_WIDTH-1:0] out
);

  
  reg [DATA_WIDTH-2:0] corr = 0;
  reg sign_buf;  

  reg [DATA_WIDTH-1:0] buffer_in;
  wire [DATA_WIDTH-1:0] buffer_out;
  reg [DATA_WIDTH-1:0] sub = 0;

  // fifo signals
  wire empty, full;
  reg read, write;

  initial begin
    write <= 1;
    read  <= 0;
    sub   <= 0;
    sign_buf <= 1;
    corr  <= 16'h0000;
  end

  fifo #(
    .ADDRESS_WIDTH(CHANNELS),
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

  // initialise
  //write = 1;

  // sync logic
  always @ (posedge clk)
  begin
    // update fifo
    buffer_in <= in;
    
    // initial 
    if (!full) begin
      sub <= in;
    end
    else begin
      read = 1;
      
      sub  = (buffer_out - in);
    end
    // output
    corr <= sub[DATA_WIDTH-1] ? 
      ((~sub[DATA_WIDTH-2:0]) ^ corr) : 
      (( sub[DATA_WIDTH-2:0]) ^ corr) ;
    sign_buf <= sub[DATA_WIDTH-1];
    //out <= {sign_buf, corr};
  end

  assign out = {sign_buf, corr};
  //assign out = 16'h5;

endmodule
