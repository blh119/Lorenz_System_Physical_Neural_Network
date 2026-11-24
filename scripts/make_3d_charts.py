"""Make 3D x-y-z charts of the Lorenz RK4 reference trajectory, t in [0, 20].

Outputs: lorenz_3d_static.png, lorenz_xyz_vs_time.png, lorenz_3d_motion.mp4 (needs ffmpeg), lorenz_3d_motion.gif,
lorenz_3d_interactive.html. Requires numpy, pandas, matplotlib, plotly.
Usage: python make_3d_charts.py --csv <reference.csv> --out <folder>
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import animation
import argparse, os
ap=argparse.ArgumentParser(description="3D Lorenz charts (static PNG, MP4/GIF animation, interactive HTML) from an RK4 reference CSV with columns t,x_pos,y_pos,z_pos.")
ap.add_argument("--csv",default="/mnt/project-files/results/horizon/pred_tm_T20_w0.5cold_s99.csv")
ap.add_argument("--out",default="/mnt/project-files/results/3d/")
a=ap.parse_args(); OUT=os.path.join(a.out,""); os.makedirs(OUT,exist_ok=True)
df=pd.read_csv(a.csv,usecols=["t","x_pos","y_pos","z_pos"])
d=df.iloc[::20].reset_index(drop=True)   # dt = 0.002
t,x,y,z=[d[c].values for c in ["t","x_pos","y_pos","z_pos"]]
# wing flips from full-res data
X=df.x_pos.values; idx=np.where(np.sign(X[1:])!=np.sign(X[:-1]))[0]
flips=df.t.values[idx]; print("flips",np.round(flips,3))
fi=[np.argmin(abs(t-f)) for f in flips]
c=np.sqrt(8/3*27); FP=[(c,c,27),(-c,-c,27),(0,0,0)]
cmap=plt.cm.viridis
def segs(n):
    P=np.c_[x[:n],y[:n],z[:n]].reshape(-1,1,3); return np.concatenate([P[:-1],P[1:]],1)
def setup(ax):
    ax.set_xlim(-20,20);ax.set_ylim(-27,27);ax.set_zlim(0,50)
    ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
    ax.xaxis.pane.fill=ax.yaxis.pane.fill=ax.zaxis.pane.fill=False
def fixed(ax):
    for i,(a,b,cz) in enumerate(FP):
        ax.scatter(a,b,cz,marker="X",s=90,c="crimson" if i<2 else "gray",edgecolors="k",depthshade=False,zorder=5,
                   label="Fixed points C± (±8.49, ±8.49, 27)" if i==0 else ("Origin (unstable)" if i==2 else None))
# ---- static ----
fig=plt.figure(figsize=(11,8.5)); ax=fig.add_subplot(projection="3d"); setup(ax)
lc=Line3DCollection(segs(len(t)),cmap=cmap,linewidth=1.0); lc.set_array(t[:-1]); lc.set_clim(0,20); ax.add_collection(lc)
ax.scatter(x[0],y[0],z[0],s=70,c="white",edgecolors="k",depthshade=False,label="Start (1, 1, 1), t = 0")
ax.scatter(x[fi],y[fi],z[fi],s=45,c="orange",edgecolors="k",marker="o",depthshade=False,label="Wing flips (x changes sign)")
fixed(ax); ax.view_init(22,-58)
cb=fig.colorbar(lc,ax=ax,shrink=0.6,pad=0.08); cb.set_label("time t")
ax.legend(loc="upper left",fontsize=9)
ax.set_title("Lorenz attractor, RK4 reference, t ∈ [0, 20]\nσ = 10, ρ = 28, β = 8/3",fontsize=13)
fig.savefig(OUT+"lorenz_3d_static.png",dpi=200,bbox_inches="tight"); plt.close(fig)
# ---- x(t), y(t), z(t) time series with wing-flip lines ----
fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True)
for a,v,lab,col in zip(axs,[x,y,z],["x(t)","y(t)","z(t)"],["#1f77b4","#2ca02c","#d62728"]):
    a.plot(t,v,c=col,lw=1.1)
    for k,f in enumerate(flips):
        a.axvline(f,c="0.35",ls="--",lw=0.9,label="Wing flip (x changes sign)" if k==0 else None)
    a.set_ylabel(lab); a.grid(alpha=0.3)
    if lab!="z(t)": a.axhline(0,c="0.6",lw=0.6)
for f in flips: axs[0].text(f,1.01,f"{f:.3f}",transform=axs[0].get_xaxis_transform(),rotation=90,va="bottom",ha="center",fontsize=8,color="0.25")
axs[0].legend(loc="lower left",fontsize=9)
axs[-1].set_xlabel("time t"); axs[-1].set_xlim(0,20)
axs[0].set_title("Lorenz system, RK4 reference, t ∈ [0, 20]   (σ = 10, ρ = 28, β = 8/3)",fontsize=13,pad=34)
fig.tight_layout(); fig.savefig(OUT+"lorenz_xyz_vs_time.png",dpi=200); plt.close(fig)
# ---- animation ----
fig=plt.figure(figsize=(9,7.5)); ax=fig.add_subplot(projection="3d"); setup(ax); fixed(ax)
ax.legend(loc="upper left",fontsize=8)
lc=Line3DCollection([],cmap=cmap,linewidth=1.1); lc.set_clim(0,20); ax.add_collection(lc)
head,=ax.plot([],[],[],"o",c="red",ms=7,mec="k")
ax.view_init(22,-58)  # fixed camera, same view as the static PNG
cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,20),cmap=cmap),ax=ax,shrink=0.55,pad=0.08); cb.set_label("time t")
title=ax.set_title("")
NF=400; ends=np.linspace(2,len(t),NF).astype(int)
def upd(k):
    n=ends[k]; s=segs(n); lc.set_segments(s); lc.set_array(t[:n-1])
    head.set_data_3d([x[n-1]],[y[n-1]],[z[n-1]])
    nf=int((flips<=t[n-1]).sum())
    title.set_text(f"Lorenz system, RK4 reference   t = {t[n-1]:5.2f}   wing flips: {nf}")
    return lc,head,title
an=animation.FuncAnimation(fig,upd,frames=NF,blit=False)
an.save(OUT+"lorenz_3d_motion.mp4",writer=animation.FFMpegWriter(fps=20,bitrate=3000),dpi=120)
an.save(OUT+"lorenz_3d_motion.gif",writer=animation.PillowWriter(fps=20),dpi=70)
plt.close(fig)
# ---- interactive html ----
import plotly.graph_objects as go
dd=df.iloc[::50]  # dt=0.005
f=go.Figure()
f.add_trace(go.Scatter3d(x=dd.x_pos,y=dd.y_pos,z=dd.z_pos,mode="lines",line=dict(color=dd.t,colorscale="Viridis",width=4,colorbar=dict(title="t")),
  name="trajectory",hovertemplate="t=%{customdata:.3f}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}",customdata=dd.t))
f.add_trace(go.Scatter3d(x=[p[0] for p in FP[:2]],y=[p[1] for p in FP[:2]],z=[27,27],mode="markers",marker=dict(size=6,color="crimson",symbol="x"),name="fixed points C±"))
f.add_trace(go.Scatter3d(x=x[fi],y=y[fi],z=z[fi],mode="markers",marker=dict(size=4,color="orange"),name="wing flips",text=[f"t={v:.3f}" for v in flips]))
f.update_layout(title="Lorenz attractor, RK4 reference, t ∈ [0, 20] (drag to rotate)",scene=dict(xaxis_title="x",yaxis_title="y",zaxis_title="z"),height=800)
f.write_html(OUT+"lorenz_3d_interactive.html",include_plotlyjs=True)
