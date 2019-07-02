// from https://embeddedthoughts.com/2016/07/13/fifo-buffer-using-block-ram-on-a-xilinx-spartan-3-fpga/
`timescale 1ns / 1ns
module sync_dual_port_ram
	
	
	#( parameter ADDRESS_WIDTH = 12, // number of words in ram
                     DATA_WIDTH    =  8  // number of bits in word
	 )
	
	// IO ports
	(
		input wire clk, write_en,                                        // signal to enable synchronous write
		input [ADDRESS_WIDTH-1:0] read_address, write_address, // inputs for dual port addresses
		input [DATA_WIDTH-1:0] write_data_in,                  // input for data to write to ram
		output reg [DATA_WIDTH-1:0] read_data_out                  // outputs for dual data ports
	);
	
	// internal signal declarations
	reg [DATA_WIDTH-1:0] ram [2**ADDRESS_WIDTH-1:0]; 
	reg [DATA_WIDTH-1:0] d0; 

	// synchronous write and address update
        // PORT A
  	always @(posedge clk)
		begin
		if (write_en)  							 // if write enabled
		   ram[write_address] <= write_data_in; // write data to ram and write_address 
		    
                //read_address_reg  <= read_address;      // store read_address to reg
	        d0 <= ram[read_address];
	        read_data_out <= d0;
                end
	
        // PORT B
        //always @(posedge clk)
	//	begin
	//        read_data_out  <= ram[read_address];
        //        end
	

	// assignments for two data out ports
	//assign read_data_out  = ram[read_address];
	//assign write_data_out = ram[write_address_reg];
endmodule
