import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
def newton_iteration_finescale(F_Jacobian, Ts, adjs, p, s, time_step, wells,all_ids, rel_tol=1e-3):
    pressure = p.copy()
    swns = s.copy()
    swn1s = s.copy()
    converged=False
    count=0
    dt=time_step
    # data_impress['swn1s']=data_impress['swns'].copy()
    # all_ids=GID_0
    # not_prod=np.setdiff1d(all_ids,wells['all_wells'])
    while not converged:
        swns[wells['ws_inj']]=1
        J, q=get_jacobian_matrix(Ts, adjs, swns, swn1s, time_step, wells, F_Jacobian, all_ids, p)
        # Ts, adjs, Swns, Swn1s, F_Jacobian, ID_vol
        # J=FIM.J
        # q=FIM.q

        sol=-linalg.spsolve(J, q)
        n=int(len(q)/2)

        pressure+=sol[0:n]
        swns+=sol[n:]
        swns[wells['ws_inj']]=1
        # converged=max(abs(sol[n:][not_prod]))<rel_tol
        print(max(abs(sol)),max(abs(sol)),'fs')
        count+=1
        if count>20:
            print('excedded maximum number of iterations finescale')
            return False, count, pressure, swns
    # saturation[wells['ws_prod']]=saturation[wells['viz_prod']].sum()/len(wells['viz_prod'])
    return True, count, pressure, swns

def get_jacobian_matrix(Ts, Adjs, Swns, Swn1s, time_step, wells, F_Jacobian, ID_vol, p):
    # Ts, adjs, swns, swn1s, time_step, wells, F_Jacobian
    n=len(ID_vol)
    count=0
    # Swns=self.swns
    # Swn1s=self.swn1s
    Swns[Swns<0]=0
    Swns[Swns>1]=1
    # Swn1s[Swn1s<0]=0
    # Swn1s[Swn1s>1]=1
    # ID_vol=self.ids
    lines=[]
    cols=[]
    data=[]
    lines.append(ID_vol)
    cols.append(n+ID_vol)
    data.append(F_Jacobian.c_o(0.3,np.repeat(time_step,n)))
    # J[ID_vol][n+ID_vol]+=float(F_Jacobian().c_o.subs({Dx:self.Dx, Dy:self.Dy, phi:0.3, Dt:self.dt}))
    lines.append(n+ID_vol)
    cols.append(n+ID_vol)
    data.append(F_Jacobian.c_w(0.3,np.repeat(time_step,n)))
    # J[n+ID_vol][n+ID_vol]+=float(F_Jacobian().c_w.subs({Dx:self.Dx, Dy:self.Dy, phi:0.3, Dt:self.dt}))
    linesq=[]
    dataq=[]
    linesq.append(ID_vol)
    dataq.append(F_Jacobian.acum_o(0.3,time_step,Swns,Swn1s))
    # q[ID_vol]+=float(F_Jacobian().acum_o.subs({Dx:self.Dx, Dy:self.Dy, phi:0.3, Dt:self.dt, Sw:Swns[count], Swn:Swn1s[count]}))
    linesq.append(n+ID_vol)
    dataq.append(F_Jacobian.acum_w(0.3,time_step,Swns,Swn1s))
    # q[n+ID_vol]+=float(F_Jacobian().acum_w.subs({Dx:self.Dx, Dy:self.Dy, phi:0.3, Dt:self.dt, Sw:Swns[count], Swn:Swn1s[count]}))
    # Adjs=np.array(self.adjs)
    adj0=np.array(Adjs[:,0])
    adj1=np.array(Adjs[:,1])
    ids0=ID_vol[adj0]
    ids1=ID_vol[adj1]
    ID_vol=ids0
    id_j=ids1
    swns0=Swns[ids0]
    swns1=Swns[ids1]
    press0=p[adj0]
    press1=p[adj1]
    pf0=press0
    pf1=press1
    up0=pf0>pf1
    up1=pf0<=pf1
    nfi=len(Adjs)
    swf=np.zeros(nfi)
    swf[up0]=swns0[up0]
    swf[up1]=swns1[up1]
    id_up=np.zeros(nfi,dtype=np.int32)
    id_up[up0]=ids0[up0]
    id_up[up1]=ids1[up1]
    # Ts=self.Ts

    J00=F_Jacobian.J[0][0](Ts,swf)
    # J00=float(self.F_Jacobian[0][0].subs({T:1, Sw:swf}))
    J01=F_Jacobian.J[0][1](Ts,swf, pf0, pf1)
    # J01=float(self.F_Jacobian[0][1].subs({T:1, Sw:swf, p_i:pv, p_j:pj}))
    J10=F_Jacobian.J[1][0](Ts,swf)
    # J10=float(self.F_Jacobian[1][0].subs({T:1, Sw:swf}))
    J11=F_Jacobian.J[1][1](Ts,swf, pf0, pf1)
    # J11=float(self.F_Jacobian[1][1].subs({T:1, Sw:swf, p_i:pv, p_j:pj}))
    linesq.append(ID_vol)
    dataq.append(-F_Jacobian.F_o(Ts,swf, pf0, pf1))
    linesq.append(id_j)
    dataq.append(-F_Jacobian.F_o(Ts,swf, pf1, pf0))
    # q[ID_vol]-=float(F_Jacobian().F_o.subs({T:1.0, Sw:Swns1[count_fac], p_i:pv, p_j:pj}))
    linesq.append(n+ID_vol)
    dataq.append(-F_Jacobian.F_w(Ts,swf, pf0, pf1))
    linesq.append(n+id_j)
    dataq.append(-F_Jacobian.F_w(Ts,swf, pf1, pf0))
    # q[n+ID_vol]-=float(F_Jacobian().F_w.subs({T:1.0, Sw:Swns1[count_fac], p_i:pv, p_j:pj}))
    lines.append(ID_vol)
    cols.append(ID_vol)
    data.append(-J00)
    lines.append(id_j)
    cols.append(id_j)
    data.append(-J00)
    # J[ID_vol][ID_vol]-=J00
    lines.append(ID_vol)
    cols.append(id_j)
    data.append(J00)
    lines.append(id_j)
    cols.append(ID_vol)
    data.append(J00)
    # J[ID_vol][id_j]+=J00
    lines.append(n+ID_vol)
    cols.append(ID_vol)
    data.append(-J10)
    lines.append(n+id_j)
    cols.append(id_j)
    data.append(-J10)
    # J[n+ID_vol][ID_vol]-=J10
    lines.append(n+ID_vol)
    cols.append(id_j)
    data.append(J10)
    lines.append(n+id_j)
    cols.append(ID_vol)
    data.append(J10)
    # J[n+ID_vol][id_j]+=J10
    lines.append(ID_vol)
    cols.append(n+id_up)
    data.append(-J01)
    lines.append(id_j)
    cols.append(n+id_up)
    data.append(J01)
    # J[ID_vol][n+id_up]-=J01
    lines.append(n+ID_vol)
    cols.append(n+id_up)
    data.append(-J11)
    lines.append(n+id_j)
    cols.append(n+id_up)
    data.append(J11)
    # J[n+ID_vol][n+id_up]-=J11
    lines=np.concatenate(lines)
    cols=np.concatenate(cols)
    data=np.concatenate(data)
    linesq=np.concatenate(linesq)
    dataq=np.concatenate(dataq)
    q=np.bincount(linesq, weights=dataq)
    lines, cols, data, q = apply_BC(lines, cols, data, q, wells)
    J=sp.csc_matrix((data,(lines,cols)),shape=(2*n,2*n))
    return(J, q)

def apply_BC(lines, cols, data, q, wells):
        n=int(len(q)/2)
        q[wells['ws_p']]=0
        q[wells['ws_inj']+n]=0
        if (wells['count']==0) and (len(wells['values_q'])>0):
            q[wells['ws_q']]+=wells['values_q']
        for l in wells['ws_p']:
            data[lines==l]=0
            lines=np.append(lines,l)
            cols=np.append(cols,l)
            data=np.append(data,1)
        for l in np.setdiff1d(wells['ws_inj'],wells['ws_q']):
            data[lines==l+n]=0
            lines=np.append(lines,l+n)
            cols=np.append(cols,l+n)
            data=np.append(data,1)

        return lines, cols, data, q
