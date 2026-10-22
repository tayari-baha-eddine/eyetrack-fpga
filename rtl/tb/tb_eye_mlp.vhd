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

  -- Signaux d'observation pour GTKWave
  signal obs_d1 : std_logic;
  signal obs_d2 : std_logic;
begin
  clk <= not clk after 5 ns;

  -- On observe les signaux internes d1 et d2
  obs_d1 <= << signal .tb_eye_mlp.dut.d1 : std_logic >>;
  obs_d2 <= << signal .tb_eye_mlp.dut.d2 : std_logic >>;

  dut : entity work.eye_mlp_top
    port map (clk => clk, rst => rst, start => start, pixels => pixels, done => done, result => result);

  stim : process
    file f : text;
    variable l : line;
    variable v, e0, e1 : integer;
    variable n, cycles, max_cycles : integer := 0;
    -- Variables pour la trace des cycles
    variable start_cycle, d1_cycle, d2_cycle : integer := 0;
  begin
    wait for 30 ns;
    wait until rising_edge(clk);
    rst <= '0';
    wait until rising_edge(clk);
    file_open(f, VEC_FILE, read_mode);
    
    -- Pour la démonstration, on ne teste que le premier vecteur
    -- (pour voir la trace exacte des cycles sans être noyé sous 106 lignes)
    readline(f, l);
    for i in 0 to 255 loop
      read(l, v);
      pixels(i) <= to_signed(v, 32);
    end loop;
    read(l, e0);
    read(l, e1);
    
    wait until rising_edge(clk);
    start_cycle := 0;
    start <= '1';
    wait until rising_edge(clk);
    start <= '0';
    
    cycles := 1;
    loop
      wait until rising_edge(clk);
      cycles := cycles + 1;
      
      -- On enregistre le cycle exact où chaque couche termine
      if obs_d1 = '1' and d1_cycle = 0 then 
        d1_cycle := cycles; 
      end if;
      if obs_d2 = '1' and d2_cycle = 0 then 
        d2_cycle := cycles; 
      end if;
      
      exit when done = '1';
      assert cycles < 100000 report "timeout : done jamais recu" severity failure;
    end loop;
    
    -- On affiche la preuve absolue
    report "TRACE CYCLES : d1=" & integer'image(d1_cycle) & 
           " | d2=" & integer'image(d2_cycle) & 
           " | done=" & integer'image(cycles);
           
    if to_integer(result(0)) /= e0 or to_integer(result(1)) /= e1 then
      report "ECART au vecteur " & integer'image(n)
        & " : VHDL=(" & integer'image(to_integer(result(0))) & "," & integer'image(to_integer(result(1)))
        & ") attendu=(" & integer'image(e0) & "," & integer'image(e1) & ")"
        severity failure;
    end if;
    
    finish;
  end process;
end architecture;
