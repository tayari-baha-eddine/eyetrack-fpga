from pathlib import Path
import numpy as np
from src.quantization.quantized_reference import load_params

NL = chr(10)


def vec(name, arr):
    flat = [int(v) for v in np.asarray(arr).flatten()]
    rows = ['    ' + ', '.join(str(v) for v in flat[i:i + 16]) for i in range(0, len(flat), 16)]
    head = '  constant %s : integer_vector(0 to %d) := (' % (name, len(flat) - 1)
    return head + NL + (',' + NL).join(rows) + NL + '  );'


qp = load_params()
parts = ['-- Fichier genere par export_vhdl_params.py : ne pas editer a la main',
         'package eye_mlp_params is']
for k in (1, 2, 3):
    parts.append(vec('W%d' % k, qp['W%d' % k]))
    parts.append(vec('B%d' % k, qp['b%d' % k]))
for k in (1, 2):
    parts.append('  constant M%d : integer := %d;' % (k, int(qp['M%d' % k])))
    parts.append('  constant S%d : integer := %d;' % (k, int(qp['S%d' % k])))
parts.append('end package;')
Path('rtl/top').mkdir(parents=True, exist_ok=True)
Path('rtl/top/eye_mlp_params.vhd').write_text(NL.join(parts) + NL)
print('Ecrit : rtl/top/eye_mlp_params.vhd')
