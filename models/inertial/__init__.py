from .mlp import mlp
from .gru import gru
from .lstm import lstm
from .cnn1d import cnn1d
from .bigru import bigru
from .bilstm import bilstm

from .cnn_bilstm import cnn_bilstm
from .res_bilstm import res_bilstm
# from .tcn import tcn
# from .dilated_tcn import dilated_tcn
# from .tri_attention import tri_attention

__all__ = ['mlp',
           'gru',
           'lstm',
           'cnn1d',
           'bilstm',
           'bigru',
           'cnn_bilstm',
           'res_bilstm',
           # 'tcn',
           # 'dilated_tcn',
           # 'tri_attention',
           ]
