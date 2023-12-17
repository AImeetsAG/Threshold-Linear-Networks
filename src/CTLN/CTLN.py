import numpy as np
from scipy.integrate import odeint
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import itertools
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import networkx as nx
from IPython.display import Image, display

class graphtodynamics:
  def __init__(self, M, X0,theta, final_time, t_ticks=10000 ):
    self.M = M
    self.X0= X0
    self.theta= theta
    self.final_time = final_time
    self.t_ticks= t_ticks

  def print_matrix(self):
    print("M= ")
    display (np.array(self.M).reshape(len(self.M),len(self.M)))
    print("W= ")
    display(self.adj_to_network())

  def get_graph(self):
    G= nx.DiGraph(np.array(np.array(self.M).transpose()))
    fig, ax = plt.subplots(figsize=(4,4)) 
    nx.draw(G, with_labels= True, node_color=range(len(self.M)), pos=nx.kamada_kawai_layout(G), cmap = matplotlib.colors.ListedColormap(plt.cm.tab10.colors[:len(self.M)]), ax=ax)
    plt.show()

  def relu(self,x):
    return np.array([max(0.0,a) for a in x])
    
  def relu_ar(self,x):
    return np.array([self.relu(a) for a in x])

  def deriv(self, X, t, Mat):
    return np.sum([-X,self.relu(np.sum([np.dot(Mat, X),self.theta*np.ones(Mat.shape[0])],axis=0))],axis=0) # here theta is taken
# to be the constant vector of ones.

  def adj_to_network(self,delta=0.5,epsilon=0.25): # input value of delta and epsilon here only
    row_len=len(self.M)
    col_len=len(self.M[0])
    W=np.zeros((row_len,col_len))
    for i in range(len(self.M)):
        for j in range(len(self.M[0])):
            if i!=j and self.M[i][j]==0:
                W[i][j]= -1-delta
            elif i!=j and self.M[i][j]==1:
                W[i][j]= -1+epsilon
            else:
                W[i][j]=0
    return W

  def solution(self):
    Mat=self.adj_to_network()
    t= np.linspace(0,self.final_time,self.t_ticks)
    sol= odeint(self.deriv, self.X0, t, args=(Mat,))
    return sol


  def plot_dynamics(self):
    t= np.linspace(0,self.final_time,self.t_ticks)
    fig, ax = plt.subplots(figsize=(20,10)) 
    for i in range(len(self.M)):
      plt.plot(t,self.solution()[:,i],label='i = %d' % i)
    plt.xlabel("time", fontsize=15)
    plt.ylabel("firing rate", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.xlim(0,self.final_time)
    #plt.ylim(0,1.5)
    plt.legend(loc= 'upper right', shadow=True)
    plt.grid()
    plt.show()
    
    
  def plot_2d_vector(self,k=0.25):
    r=8 # max limit for x and y in the plot
    x,y = np.meshgrid(np.arange(0, r, k),
                  np.arange(0, r, k))
    W= self.adj_to_network()
    u= -x + self.relu_ar(W[0][0]*x+ W[0][1]*y+self.theta)
    v= -y + self.relu_ar(W[1][0]*x+ W[1][1]*y+self.theta)
    # Create quiver figure
    fig = ff.create_quiver(x, y, u, v,
                           scale=.1,
                           arrow_scale=.2,
                           name='quiver',
                           line_width=1)
    fig.add_trace(
    go.Scatter(x=[-self.theta/W[1][0],-self.theta/W[1][0]], y=[0,r], name="slope", line_shape='linear'))
    
    fig.add_trace(
    go.Scatter(y=[-self.theta/W[0][1],-self.theta/W[0][1]], x=[0,r], name="slope", line_shape='linear'))

    fig.show()
    

  def plot_3d_vector(self):
    x= np.linspace(0,10,5)
    y= np.linspace(0,10,5)
    z= np.linspace(0,10,5)
    new_list = np.array([item for item in itertools.product(x,y,z)])
    Mat= self.adj_to_network()
    uvw= np.array([np.sum([-new_list[i],self.relu(np.sum([np.dot(Mat, new_list[i]),self.theta*np.ones(Mat.shape[0])],axis=0))],axis=0) for i in range(len(new_list))])
    fig = go.Figure(data = go.Cone(
    x=new_list[:,0],
    y=new_list[:,1],
    z=new_list[:,2],
    u=uvw[:,0],
    v=uvw[:,1],
    w=uvw[:,2],
    colorscale='Emrld',
    sizemode="absolute",
    sizeref=20))
    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))

    fig.show()

    
  def show_3danimation(self):
        solution=self.solution()
        # Generate curve data
        t= np.linspace(0,self.final_time, self.t_ticks)
        f=4
        N=1000//f
        x = solution[:,0]
        y = solution[:,1]
        z = solution[:,2]
        xm = np.min(x) - 1.5
        xM = np.max(x) + 1.5
        ym = np.min(y) - 1.5
        yM = np.max(y) + 1.5
        zm = np.max(z) - 1.5
        zM = np.max(z) + 1.5
        xx = solution[:,0][::f] #how fast? replace by 2 to get slower or bigger than 3 for faster
        yy = solution[:,1][::f] #how fast? replace by 2 to get slower or bigger than 3 for faster
        zz = solution[:,2][::f]

        # Create figure
        fig = go.Figure(
            data=[go.Scatter3d (x=x, y=y,z=z,
                             mode="lines",
                              marker=dict(
        size=12,
        color='rgba(255, 15, 10, 0.85)',
        line=dict(
            color='blue',
            width=2
        )
    ) ),
                  go.Scatter3d (x=x, y=y,z=z,
                             mode="lines",
                              marker=dict(
        size=12,
        color='rgba(255, 15, 10, 0.85)',
        line=dict(
            color='blue',
            width=2
        )
    ))],
            layout=go.Layout(
                scene= dict( xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
                yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
                zaxis=dict(range=[zm, zM], autorange=False, zeroline=False)),
                title_text="Trajectory in Phase Space", hovermode="closest",
                updatemenus=[dict(type="buttons",
                                  buttons=[dict(label="Play",
                                                method="animate",
                                                args=[None]),dict(label="Reset",
                                                method="animate",
                                                args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}]) ])]),  


            frames=[go.Frame(
                data=[go.Scatter3d(
                    x=[xx[k]],
                    y=[yy[k]],
                    z=[zz[k]],
                    mode="markers",
                    marker=dict(color="red", size=10))])

                for k in range(N)]
        )



        fig.show()
  
  def show_2danimation(self):
    solution=self.solution()
    # Generate curve data
    t= np.linspace(0,5, 150)
    x = solution[:,0]
    y = solution[:,1]
    #z = solution[:,2]
    f=3
    N=1000//f

    xm = np.min(x) - 1.5
    xM = np.max(x) + 1.5
    ym = np.min(y) - 1.5
    yM = np.max(y) + 1.5
    xx = solution[:,0][::f] #how fast? replace by 2 to get slower or bigger than 3 for faster
    yy = solution[:,1][::f] #how fast? replace by 2 to get slower or bigger than 3 for faster
    #zz = solution[:,2][::2]


    # Create figure
    fig = go.Figure(
        data=[go.Scatter(x=x, y=y,
                         mode="lines",
                         line=dict(width=2, color="blue")),
              go.Scatter(x=x, y=y,
                         mode="lines",
                         line=dict(width=2, color="blue"))],
        layout=go.Layout(
            xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
            yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
            title_text="Trajectory in Phase Space", hovermode="closest",
            updatemenus=[dict(type="buttons",
                              buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None]),dict(label="Reset",
                                            method="animate",
                                            args=[[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}]) ])]),  


        frames=[go.Frame(
            data=[go.Scatter(
                x=[xx[k]],
                y=[yy[k]],
                mode="markers",
                marker=dict(color="red", size=10))])

            for k in range(N)]
    )



    fig.show()