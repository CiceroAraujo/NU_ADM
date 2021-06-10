from packs.preprocessor.finescale_mesh import get_transmissibilities_and_adjacencies
from packs.preprocessor.symbolic_calculation import symbolic_J as s_J
from packs.processor.finescale_processing import newton_iteration_finescale
import numpy as np

wells, GID_0, adjs, face_permeabilities = get_transmissibilities_and_adjacencies()
F_Jacobian=s_J()
p=np.zeros_like(GID_0).astype(np.float64)
p[wells['ws_p']]=wells['values_p']
s=p.copy()
time_step=0.0001
wells['count']=0
adm_conv, fs_iters, p, s=newton_iteration_finescale(F_Jacobian, face_permeabilities,
                            adjs, p, s , time_step, wells, GID_0)
import pdb; pdb.set_trace()
