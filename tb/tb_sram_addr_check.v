`timescale 1ns / 1ps
// Quick SRAM address mapping check
module tb_sram_addr_check;
    parameter CLK = 10;
    reg clk = 0;
    always #(CLK/2) clk = ~clk;

    // Check address mapping
    // SRAM addr format: [19:0]
    // Bank select: addr[1:0] XOR addr[9:8] (for bank conflict avoidance)
    // Word addr: addr[19:2]
    
    function [1:0] get_bank;
        input [19:0] addr;
        begin
            get_bank = addr[1:0] ^ addr[9:8];
        end
    endfunction
    
    function [17:0] get_word;
        input [19:0] addr;
        begin
            get_word = addr[19:2];
        end
    endfunction

    initial begin
        $display("SRAM Address Mapping Check");
        $display("");
        $display("Address 0x0000: bank=%0d, word=%0d", get_bank(20'h0000), get_word(20'h0000));
        $display("Address 0x0001: bank=%0d, word=%0d", get_bank(20'h0001), get_word(20'h0001));
        $display("Address 0x0002: bank=%0d, word=%0d", get_bank(20'h0002), get_word(20'h0002));
        $display("Address 0x0003: bank=%0d, word=%0d", get_bank(20'h0003), get_word(20'h0003));
        $display("");
        $display("Address 0x0010: bank=%0d, word=%0d", get_bank(20'h0010), get_word(20'h0010));
        $display("Address 0x0011: bank=%0d, word=%0d", get_bank(20'h0011), get_word(20'h0011));
        $display("Address 0x0012: bank=%0d, word=%0d", get_bank(20'h0012), get_word(20'h0012));
        $display("Address 0x0013: bank=%0d, word=%0d", get_bank(20'h0013), get_word(20'h0013));
        $display("");
        $display("For batch test:");
        $display("  W at 0x0000-0x0003 -> bank 0-3, word 0");
        $display("  X at 0x0010-0x0013 -> bank 0-3, word 4");
        $display("  Z at 0x0020-0x0023 -> bank 0-3, word 8");
        $finish;
    end
endmodule
