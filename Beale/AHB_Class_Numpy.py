import numpy as np

class AHBOptimizer():
    def __init__(self, function, gradients, x_init=None, y_init=None, c=2):
        
        self.f = function
        self.g = gradients
        scale = 1.0
        self.vars = np.zeros([2,1])
        
        if x_init is not None:
          self.vars[0,0] = x_init
        else:
          self.vars[0,0] = np.random.uniform(low=0, high=scale)
          
        if y_init is not None:
          self.vars[1,0] = y_init
        else:
          self.vars[1,0] = np.random.uniform(low=0, high=scale)
          
        print("x_init: {:.3f}".format(self.vars[0,0]))
        print("y_init: {:.3f}".format(self.vars[1,0]))
        
        self.gamma = 1.
        self.c = c
       
        # for accumulation of loss and path (w, b)
        self.z_history = []
        self.x_history = []
        self.y_history = []
     
    def func(self, variables):
      """Beale function.
     
      Args:
        variables: input data, shape: 1-rank Tensor (vector) np.array
          x: x-dimension of inputs
          y: y-dimension of inputs
       
      Returns:
        z: Beale function value at (x, y)
      """
      x, y = variables[0,0], variables[1,0]
      z = self.f(x, y)
      return z
   
    def gradients(self, variables):
      """Gradient of Beale function.
     
      Args:
        variables: input data, shape: 1-rank Tensor (vector) np.array
          x: x-dimension of inputs
          y: y-dimension of inputs
       
      Returns:
        grads: [dx, dy], shape: 1-rank Tensor (vector) np.array
          dx: gradient of Beale function with respect to x-dimension of inputs
          dy: gradient of Beale function with respect to y-dimension of inputs
      """
      x, y = variables[0,0], variables[1,0]
      grads = self.g(x, y)
      grads = np.reshape(grads, (2,1))
      return grads
  
    def weights_update(self, params, grads, prev_p, prev_g, time):
        
        # print('params')
        # print(params)
        # print('prev_p')
        # print(prev_p)
        # print('grads')
        # print(grads)
        # print('prev_g')
        # print(prev_g)
        
        dx = params - prev_p
        dg = grads - prev_g
        
        # print('dx')
        # print(dx)
        # print('dg')
        # print(dg)
        # dx=np.reshape(dx, (2,1))
        # dg=np.reshape(dg, (2,1))
        
        P = np.matmul(dg.T,dg)/np.matmul(dx.T,dx)
        Lh = self.c*np.sqrt(P)
        
        # print('P')
        # print(P)
        # print('Lh')
        # print(Lh)
        
        xb = dg - Lh*dx
        
        # print('xb')
        # print(xb)
        
        p = np.matmul(xb.T,xb)/np.matmul(dx.T,dx)
        lh = np.sqrt(p)
        
        alfah = 4/((np.sqrt(Lh) + np.sqrt(lh))**2)
        betah = ((np.sqrt(Lh) - np.sqrt(lh))/(np.sqrt(Lh) + np.sqrt(lh)))**2
        
        # self.gamma = alfah
        # self.beta = betah
        
        self.prev_p = self.vars
        self.prev_g = self.gradients(self.vars)
        
        self.vars = self.vars - alfah*grads + betah*dx
        # print('vars')
        # print(self.vars)
        
    def history_update(self, z, x, y):
        """Accumulate all interesting variables
        """
        self.z_history.append(z)
        self.x_history.append(x)
        self.y_history.append(y)
            
    def train(self, max_steps):
      self.z_history = []
      self.x_history = []
      self.y_history = []
      
      self.prev_p = 1*np.random.rand(2,1) #0*np.ones([2])
      self.prev_g = self.gradients(self.prev_p)
      
      pre_z = 0.0
      print("steps: {}  z: {:.6f}  x: {:.5f}  y: {:.5f}".format(0, self.func(self.vars), self.x, self.y))
     
      file = open('adahb.txt', 'w')
      file.write("{:.5f}  {:.5f}\n".format(self.x, self.y))
     
      for step in range(max_steps):
        self.z = self.func(self.vars)
        self.history_update(self.z, self.x, self.y)
  
        self.grads = self.gradients(self.vars)
        self.weights_update(self.vars, self.grads, self.prev_p, self.prev_g, step+1)
        file.write("{:.5f}  {:.5f}\n".format(self.x, self.y))
        
        
        if (step+1) % 100 == 0:
          print("steps: {}  z: {:.6f}  x: {:.5f}  y: {:.5f}  dx: {:.5f}  dy: {:.5f}".format(step+1, self.func(self.vars), self.x, self.y, self.dx, self.dy))
         
        if np.abs(self.z) < 1e-5:
          print("Enough convergence")
          print("steps: {}  z: {:.6f}  x: {:.5f}  y: {:.5f}".format(step+1, self.func(self.vars), self.x, self.y))
          self.z = self.func(self.vars)
          self.history_update(self.z, self.x, self.y)
          break
         
        pre_z = self.z
        
        
      file.close()
  
      self.x_history = np.array(self.x_history)
      self.y_history = np.array(self.y_history)
      self.path = np.concatenate((np.expand_dims(self.x_history, 1), np.expand_dims(self.y_history, 1)), axis=1).T
       
    @property
    def x(self):
      return self.vars[0,0]
   
    @property
    def y(self):
      return self.vars[1,0]
   
    @property
    def dx(self):
      return self.grads[0,0]
   
    @property
    def dy(self):
      return self.grads[1,0]
  