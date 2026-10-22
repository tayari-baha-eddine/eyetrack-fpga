library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.mlp_pkg.all;
use work.eye_mlp_params.all;

entity eye_mlp_top is
  port (
    clk    : in  std_logic;
    rst    : in  std_logic;
    start  : in  std_logic;
    pixels : in  act_array(0 to 255);
    done   : out std_logic;
    result : out act_array(0 to 1)
  );
end entity;

architecture rtl of eye_mlp_top is
  signal a1     : act_array(0 to 15);
  signal a2     : act_array(0 to 7);
  signal d1, d2 : std_logic;
begin
  l1 : entity work.dense_layer
    generic map (N_IN => 256, N_OUT => 16, W => W1, B => B1, M => M1, S => S1, RAW_OUT => false)
    port map (clk => clk, rst => rst, start => start, x => pixels, done => d1, y => a1);

  l2 : entity work.dense_layer
    generic map (N_IN => 16, N_OUT => 8, W => W2, B => B2, M => M2, S => S2, RAW_OUT => false)
    port map (clk => clk, rst => rst, start => d1, x => a1, done => d2, y => a2);

  l3 : entity work.dense_layer
    generic map (N_IN => 8, N_OUT => 2, W => W3, B => B3, M => 0, S => 0, RAW_OUT => true)
    port map (clk => clk, rst => rst, start => d2, x => a2, done => done, y => result);
end architecture;

