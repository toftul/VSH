from  VSH_core import *
# for interactive plots
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def vectorFieldCoverter_sph2car(Asph, r, theta, phi):    
    # A^cart = R(θ, φ) A^sph
    # https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
    st, sp = np.sin(theta), np.sin(phi)
    ct, cp = np.cos(theta), np.cos(phi)

    R_rot = np.array([
        [st*cp, ct*cp, -sp],
        [st*sp, ct*sp,  cp],
        [   ct,   -st,   0]
    ])
    
    return R_rot @ Asph

r = 10
m, n = 0, 1

# Neven, Meven, Nodd, Modd
mode_type = "Meven"

plot_title = ""

if mode_type == "Neven":
    mode_func = VSH_Nemn
    plot_title = "Ne" + str(m) + str(n)
elif mode_type == "Meven":
    mode_func = VSH_Memn
    plot_title = "Me" + str(m) + str(n)
elif mode_type == "Nodd":
    mode_func = VSH_Nomn
    plot_title = "No" + str(m) + str(n)
elif mode_type == "Modd":
    mode_func = VSH_Momn
    plot_title = "Mo" + str(m) + str(n)
else:
    print("ERROR")

    
# ---------------------------------------------------------
# DATA FOR THE "FLOWER"
# ---------------------------------------------------------
phi, theta = np.mgrid[0:2*np.pi:200j, 0:np.pi:200j]

xx = np.cos(phi) * np.sin(theta)
yy = np.sin(phi) * np.sin(theta)
zz = np.cos(theta)


VSH_r = mode_func(m, n, r, theta, phi)[0]
VSH_p = mode_func(m, n, r, theta, phi)[1]
VSH_t = mode_func(m, n, r, theta, phi)[2]

rr = VSH_r**2 + VSH_p**2 + VSH_t**2

rr = rr / np.max(rr)

# ---------------------------------------------------------
# DATA FOR CONES
# ---------------------------------------------------------
num_pts = 300
indices = np.arange(0, num_pts, dtype=float) + 0.5

# equally distributed points on a sphere
# https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
ppp = pi * (1 + 5**0.5) * indices 
ttt = np.arccos(1 - 2*indices/num_pts)

SIZE_LINEAR = num_pts

xxx = np.zeros(SIZE_LINEAR)
yyy = np.zeros(SIZE_LINEAR)
zzz = np.zeros(SIZE_LINEAR)
AAx = np.zeros(SIZE_LINEAR)
AAy = np.zeros(SIZE_LINEAR)
AAz = np.zeros(SIZE_LINEAR)

AAabs = np.zeros(SIZE_LINEAR)

for index in range(SIZE_LINEAR):
    pp = ppp[index]
    tt = ttt[index]
    
    xxx[index] = np.cos(pp) * np.sin(tt)
    yyy[index] = np.sin(pp) * np.sin(tt)
    zzz[index] = np.cos(tt)

    #print(np.array([xxx[index], yyy[index], zzz[index]]))
    AAx[index] = vectorFieldCoverter_sph2car(mode_func(m, n, r, tt, pp), 1, tt, pp)[0]
    AAy[index] = vectorFieldCoverter_sph2car(mode_func(m, n, r, tt, pp), 1, tt, pp)[1]
    AAz[index] = vectorFieldCoverter_sph2car(mode_func(m, n, r, tt, pp), 1, tt, pp)[2]
    AAabs[index] = np.sqrt(AAx[index]**2 + AAy[index]**2 + AAz[index]**2)


# normalization of the vector field
AAx = AAx / np.max(AAabs)
AAy = AAy / np.max(AAabs)
AAz = AAz / np.max(AAabs)
    
# ---------------------------------------------------------
# DATA FOR THE SPIN
# ---------------------------------------------------------


fig = go.Figure()


fig.add_trace(
    go.Surface(
        x=rr*xx, y=rr*yy, z=rr*zz, surfacecolor=rr,
        colorscale='viridis',
    )
)

fig.add_trace(
    go.Cone(
        x=xxx,
        y=yyy,
        z=zzz,
        u=AAx,
        v=AAy,
        w=AAz,
        #colorscale='deep', # https://plotly.com/python/builtin-colorscales/
        sizemode="absolute",
        sizeref=0.1,
        showscale=False,
        opacity=1,
    )
)

'''
fig.add_trace(
    go.Streamtube(
        x = xxx,
        y = yyy,
        z = zzz,
        u = AAx,
        v = AAy,
        w = AAz,
        #starts = dict(
        #    x = [80] * 16,
        #    y = [20,30,40,50] * 4,
        #    z = [0,0,0,0,5,5,5,5,10,10,10,10,15,15,15,15]
        #),
        sizeref = 0.003,
        #colorscale = 'Portland',
        #showscale = False,
        #maxdisplayed = 3000
    )
)
'''

fig.update_layout(scene_aspectmode='cube')

fig.update_layout(
    scene = dict(
        xaxis = dict(range=[-1,1],),
        yaxis = dict(range=[-1,1],),
        zaxis = dict(range=[-1,1],),),
    width=700,
    title_text=plot_title,
    #margin=dict(r=20, l=10, b=10, t=10)
)

fig.show()
