library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.mlp_pkg.all;

entity dense_layer is
  generic (
    N_IN    : positive;
    N_OUT   : positive;
    W       : integer_vector;  -- N_OUT*N_IN poids, ligne par ligne [sortie][entree]
    B       : integer_vector;  -- N_OUT biais
    M       : integer;
    S       : natural;
    RAW_OUT : boolean := false
  );
  port (
    clk   : in  std_logic;
    rst   : in  std_logic;
    start : in  std_logic;
    x     : in  act_array(0 to N_IN - 1);
    done  : out std_logic;
    y     : out act_array(0 to N_OUT - 1)
  );
end entity;

architecture rtl of dense_layer is
  type state_t is (IDLE, RUN);
  signal state  : state_t := IDLE;
  signal o_idx  : natural range 0 to N_OUT - 1 := 0;
  signal i_idx  : natural range 0 to N_IN - 1 := 0;
  signal acc    : signed(31 downto 0) := (others => '0');
  signal y_reg  : act_array(0 to N_OUT - 1) := (others => (others => '0'));
  signal done_r : std_logic := '0';
begin
  y    <= y_reg;
  done <= done_r;

  process (clk)
    variable acc_next : signed(31 downto 0);
    variable q        : signed(63 downto 0);
    variable res      : signed(31 downto 0);
  begin
    if rising_edge(clk) then
      done_r <= '0';
      if rst = '1' then
        state <= IDLE;
        o_idx <= 0;
        i_idx <= 0;
      else
        case state is
          when IDLE =>
            if start = '1' then
              state <= RUN;
              o_idx <= 0;
              i_idx <= 0;
              acc   <= to_signed(B(B'low), 32);
            end if;
          when RUN =>
            -- MAC : acc + entree * poids
            acc_next := acc + resize(x(i_idx) * to_signed(W(W'low + o_idx * N_IN + i_idx), 8), 32);
            if i_idx = N_IN - 1 then
              if RAW_OUT then
                res := acc_next;
              else
                q := resize(acc_next * to_signed(M, 18), 64);
                if S > 0 then
                  q := q + shift_left(to_signed(1, 64), S - 1);
                end if;
                q := shift_right(q, S);
                if q < 0 then
                  res := (others => '0');
                elsif q > 127 then
                  res := to_signed(127, 32);
                else
                  res := resize(q, 32);
                end if;
              end if;
              y_reg(o_idx) <= res;
              if o_idx = N_OUT - 1 then
                state  <= IDLE;
                done_r <= '1';
              else
                o_idx <= o_idx + 1;
                i_idx <= 0;
                acc   <= to_signed(B(B'low + o_idx + 1), 32);
              end if;
            else
              i_idx <= i_idx + 1;
              acc   <= acc_next;
            end if;
        end case;
      end if;
    end if;
  end process;
end architecture;
