library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
use work.mlp_pkg.all;

entity tb_dense_small is
end entity;

architecture sim of tb_dense_small is
  constant WS : integer_vector(0 to 5) := (1, 2, 3, -1, -2, -3);
  constant BS : integer_vector(0 to 1) := (5, -5);
  signal clk   : std_logic := '0';
  signal rst   : std_logic := '1';
  signal start : std_logic := '0';
  signal x     : act_array(0 to 2) := (to_signed(10, 32), to_signed(20, 32), to_signed(30, 32));
  signal d_raw, d_q : std_logic;
  signal y_raw, y_q : act_array(0 to 1);
begin
  clk <= not clk after 5 ns;

  u_raw : entity work.dense_layer
    generic map (N_IN => 3, N_OUT => 2, W => WS, B => BS, M => 0, S => 0, RAW_OUT => true)
    port map (clk => clk, rst => rst, start => start, x => x, done => d_raw, y => y_raw);

  u_q : entity work.dense_layer
    generic map (N_IN => 3, N_OUT => 2, W => WS, B => BS, M => 16384, S => 15, RAW_OUT => false)
    port map (clk => clk, rst => rst, start => start, x => x, done => d_q, y => y_q);

  process
  begin
    wait for 25 ns;
    rst <= '0';
    wait until rising_edge(clk);
    start <= '1';
    wait until rising_edge(clk);
    start <= '0';

    -- attente robuste de d_raw (un cycle supplementaire, sans ambiguite de delta)
    while d_raw /= '1' loop
      wait until rising_edge(clk);
    end loop;

    assert to_integer(y_raw(0)) = 145 and to_integer(y_raw(1)) = -145
      report "sortie brute fausse" severity failure;
    assert to_integer(y_q(0)) = 73 and to_integer(y_q(1)) = 0
      report "requantification fausse" severity failure;

    report "dense_layer : test OK";
    finish;
  end process;
end architecture;
