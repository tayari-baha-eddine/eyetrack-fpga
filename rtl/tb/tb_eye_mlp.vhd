library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use std.env.all;
use work.mlp_pkg.all;

entity tb_eye_mlp is
  generic (
    VEC_FILE : string  := "tests/vectors/synthetic.txt";
    MAX_VEC  : natural := 1000000
  );
end entity;

architecture sim of tb_eye_mlp is
  signal clk    : std_logic := '0';
  signal rst    : std_logic := '1';
  signal start  : std_logic := '0';
  signal done   : std_logic;
  signal pixels : act_array(0 to 255) := (others => (others => '0'));
  signal result : act_array(0 to 1);
begin
  clk <= not clk after 5 ns;

  dut : entity work.eye_mlp_top
    port map (clk => clk, rst => rst, start => start, pixels => pixels, done => done, result => result);

  stim : process
    file f : text;
    variable l : line;
    variable v, e0, e1 : integer;
    variable n, cycles, max_cycles : integer := 0;
  begin
    wait for 30 ns;
    wait until rising_edge(clk);
    rst <= '0';
    wait until rising_edge(clk);
    file_open(f, VEC_FILE, read_mode);
    while not endfile(f) and n < MAX_VEC loop
      readline(f, l);
      for i in 0 to 255 loop
        read(l, v);
        pixels(i) <= to_signed(v, 32);
      end loop;
      read(l, e0);
      read(l, e1);
      wait until rising_edge(clk);
      start <= '1';
      wait until rising_edge(clk);
      start <= '0';
      cycles := 1;
      loop
        wait until rising_edge(clk);
        cycles := cycles + 1;
        exit when done = '1';
        assert cycles < 100000 report "timeout : done jamais recu" severity failure;
      end loop;
      if to_integer(result(0)) /= e0 or to_integer(result(1)) /= e1 then
        report "ECART au vecteur " & integer'image(n)
          & " : VHDL=(" & integer'image(to_integer(result(0))) & "," & integer'image(to_integer(result(1)))
          & ") attendu=(" & integer'image(e0) & "," & integer'image(e1) & ")"
          severity failure;
      end if;
      if cycles > max_cycles then
        max_cycles := cycles;
      end if;
      n := n + 1;
    end loop;
    report "OK : " & integer'image(n) & " vecteurs identiques bit a bit, latence mesuree "
      & integer'image(max_cycles) & " cycles (a quelques cycles pres)";
    finish;
  end process;
end architecture;
