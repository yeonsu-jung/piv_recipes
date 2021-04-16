import numpy as np
from matplotlib import pyplot as plt

# %%     

def plot_velocity_profile(x, v_avg,v_std, start_i, stop_i, step_i ,ax, **kwargs):
    v_array = -v_avg[start_i:stop_i:step_i,:]
    v_std_array = v_std[start_i:stop_i:step_i,:]
    y_array = y[start_i:stop_i:step_i,0]
    N = len(v_array)
    s = []
    for yy in (y_array - 55): # -55 for rearrange x axis with LE being zero.
        s.append('%.2f'%yy)

    C = 0.005
    
    for i,vv in enumerate(v_array):
        # ax.plot(vv + C*y_array[i],x[0,:],'k--')
        ax.plot(C*vv + y_array[i],x[0,:],'--',**kwargs)
        ax.plot([y_array[i],y_array[i]],[0,np.max(x)],'b-')
        ax.plot([y_array[i]+C*540,y_array[i]+C*540],[0,np.max(x)],'b--')
        for j, vvv in enumerate(vv):
            ax.arrow(y_array[i], x[0,j], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')

    ax.set_xticks(y_array)
    ax.set_xticklabels(s)

    return ax

def plot_dimensionless_velocity_profile(x, y, v_avg,v_std, start_i, stop_i, step_i ,ax, **kwargs):
    v_array = -v_avg[start_i:stop_i:step_i,:]
    v_std_array = v_std[start_i:stop_i:step_i,:]
    y_array = y[start_i:stop_i:step_i,0]
    N = len(v_array)
    s = []
    
    C = 4
    
    rho = 1e3
    mu = 1e-3

    U = np.mean(-v_avg[400:800,-1])*1e-3       

    for yy in (y_array): # -55 for rearrange x axis with LE being zero.
        Re = rho*U*(yy-55)/mu*1e-3
        s.append('%d'%Re)

    for i,vv in enumerate(v_array):
        # ax.plot(vv + C*y_array[i],x[0,:],'k--')
        # U = np.max(vv)*1e-3
        # U = 0.5       

        Re = rho*U*(y_array[i]-55)/mu*1e-3       
        # x_c = (U/(mu/rho)/((y_array[i]-55)*1e-3))**0.5*1e-3
        x_c = 1

        ax.plot(C*vv + Re,x[0,:]*x_c,'--',**kwargs)
        ax.plot([Re,Re],[0,np.max(x)*x_c],'b-')
        ax.plot([Re+C*U*1e3,Re+C*U*1e3],[0,np.max(x)*x_c],'b--')
        for j, vvv in enumerate(vv):
            ax.arrow(Re, x[0,j], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')
    
    xx = np.linspace(0,0.12,100)
    Re_x = np.linspace(0,rho*U*0.12/mu,100)
    delta = 5*((mu/rho)*xx/U)**0.5*1e3
    ax.plot(Re_x,delta,'g-')

    ax.set_xticks(rho*U*(y_array-55)/mu*1e-3)
    ax.set_xticklabels(s)
    ax.set_xlabel('Re_x')
    ax.set_ylabel('y (mm)')
    ax.axis([0,np.max(Re_x),0,2.5])

    return ax

def plot_single_profile(i,x,y,v_array,v_std_array,ax,x_cut = 0,**kw):
    C = 1        
    # y_array[i]

    rho = 1e3
    mu = 1e-3
    U = np.mean(-v_avg[400:800,-1])*1e-3       

    vv = v_array[i,:]
    ax.plot(C*vv[x_cut:],y[x_cut:],'--',**kw)
    ax.plot([0,0],[0,np.max(y)],'b-')
    ax.plot([0+C*U*1e3,C*U*1e3],[0,np.max(y)],'b--')
    for j, vvv in enumerate(vv[x_cut:]):         
        ax.arrow(0, y[j+x_cut], C*(vvv), 0, head_width=0.02, head_length=0.2, fc='k', ec='k')
        ax.arrow(0 + C*(vvv), y[j+x_cut], C*v_std_array[i,j+x_cut], 0, head_width=0.05, head_length=2, fc='r', ec='r')
        ax.arrow(0 + C*(vvv), y[j+x_cut], -C*v_std_array[i,j+x_cut], 0, head_width=0.05, head_length=2, fc='r', ec='r')

    ax.set_title('x = %.2f mm; Re_x = %d'%(x[i]-55,rho*U*(x[i]-55)/mu*1e-3))
    ax.set_xlabel('u (mm/s)')
    ax.set_ylabel('y (mm)')
    ax.axis([0,U*1.1*1e3,0,2.2])
    # plt.savefig('individual_profile.png')
    # fig.savefig('individual_profile.png',dpi=900)
    return ax

def velocity_array(x,y,ul,vl,ur,vr):
    y_lr = x[0,:]
    u_left = -vl[:,3,:]
    v_left = ul[:,3,:]

    u_right = -vr[:,-1,:]
    v_right = ur[:,-1,:]

    x_top = y2[:,0]
    u_top = -v2[:,:,0]
    v_top = -u2[:,:,0]

    x_bottom = y1[:,-3]
    u_bottom = -v1[:,:,-3]
    v_bottom = -u1[:,:,-3]
    return y_lr,u_left,v_left,u_right,v_right,x_top,u_top,v_top,x_bottom,u_bottom,v_bottom