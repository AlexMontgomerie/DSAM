import math
import json

# load config
with open('test/data/data.json','r') as f:
    config = json.load(f)

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
  test_num=config['test_num'],
  addr_width=math.ceil(math.log(int(config['channels']),2)),
  data_width=config['data_width'],
  channels=config['channels']
)

values = ''

for i in range(config['size']):
    if i == 0:
        values += '''
  data_in = 16'd{data_in};
  correct = 16'd{correct};'''.format(data_in=config['data_in'][i],correct=0)
    else:
        values += '''
  #20 data_in = 16'd{data_in};
  correct = 16'd{correct};'''.format(data_in=config['data_in'][i],correct=config['correct'][i-1])

values += '''
  #20 correct = 16'd{correct};'''.format(correct=config['correct'][config['size']-1])



footer_template = '''
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
'''

# Saving to file
with open("hw/tb/dsam_encoder_tb.v","w") as tb:
    tb.write(header_template+values+footer_template)



