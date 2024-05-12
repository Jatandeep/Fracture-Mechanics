from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
from dolfin import *
from ufl import nabla_div

set_log_level(30)
mesh1 = UnitSquareMesh(64,64)
mesh=refine(mesh1)
#### material parameters ######
c = 1e-3
k = 1e-6
Gc = 2.7
E = 210e3
nu = 0.3
#l_0 = 7.5e-3
l_0 = 0.011
mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)
print ('lamda,mu', lmbda,mu)

#### Spaces #####
W = VectorFunctionSpace(mesh,'CG',1)
u , v = TrialFunction(W),TestFunction(W)

V = FunctionSpace(mesh,'CG',1)
p , q = TrialFunction(V), TestFunction(V)


####### classes #########
#Coordinate system with bottom as reference axis:
class top(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]-1.0) < tol and on_boundary

class bottom(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]) < tol and on_boundary

class middle(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-3
        return abs(x[1]-0.5) < tol and x[0] <= 0.5
'''
# Coordinate system with crack as reference axis:
class top(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]-0.5) < tol and on_boundary

class bottom(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]+0.5) < tol and on_boundary

class middle(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]) < tol and x[0] <= 0.5
'''

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return lmbda*nabla_div(u)*Identity(2) + 2*mu*epsilon(u)
'''
kb = lmbda*(1+nu)/3.0/nu
def hist(u):
    return 0.5*kb*( tr(epsilon(u)) - abs(tr(epsilon(u))))**2.0 + ((1.0-k)*c**2.0 +k)*(0.5*kb*abs(tr(epsilon(u)))**2.0 + mu*tr(dev(epsilon(u))*dev(epsilon(u))) )

# Alternate definitions for history function:
def hist(u):
	    return  0.5*lmbda*(tr(epsilon(u)) )**2 + mu*tr(epsilon(u)*epsilon(u))
'''
kn = lmbda + mu
def hist(u):
    return 0.5*kn*( 0.5*(tr(epsilon(u)) + abs(tr(epsilon(u)))) )**2 + mu*tr(dev(epsilon(u))*dev(epsilon(u)))

####### Boundary Conditions ########

Top = top()
Bottom = bottom()
Middle = middle()

u_Lx = Expression("t",t = 0.0, degree=1)
ftx =  DirichletBC(W.sub(0),u_Lx,Top)
fty =  DirichletBC(W.sub(1),Constant(0.0),Top)

fbx =  DirichletBC(W.sub(0),Constant(0.0),Bottom)
fby =  DirichletBC(W.sub(1),Constant(0.0),Bottom)

bc_u = [ftx,fty,fbx,fby]	

bc_c = DirichletBC(V,Constant(1.0),Middle,method = 'pointwise')

######## initialization ############

u_old, u_conv, u_new = Function(W),Function(W), Function(W)
c_old, c_conv = Function(V),Function(V)
'''
####### Variational Problem ########
###Definition according to paper Borden
E_u = ((1-k)*c_old**2.0 + k)*inner(grad(v),sigma(u))*dx
E_c = ((((4.0*l_0*(1.0-k)* hist(u_new))/Gc) + 1.0)*inner(p,q) + 4.0*l_0**2.0*inner(grad(p),grad(q)) -1.0*q)*dx 
'''
###Alternate definition for E_u & E_c:
E_u =  ( pow((1-c_old),2) + 1e-6)*inner(grad(v), sigma(u))*dx
E_c = ( Gc*l_0*inner(grad(p),grad(q))+ ((Gc/l_0) + 2.*hist(u_new))*inner(p,q)- 2.*hist(u_new)*q)*dx

u = Function(W)
p = Function(V)

p_disp = LinearVariationalProblem(lhs(E_u),rhs(E_u),u,bc_u)
p_c = LinearVariationalProblem(lhs(E_c),rhs(E_c),p, bc_c)

###### solver #######
solver_disp = LinearVariationalSolver(p_disp)
solver_c = LinearVariationalSolver(p_c)

t = 1e-5
#max_load = 500e-5
max_load = 0.03
deltaT  = 1e-5
ut = 1.0

###### Displacement controlled test ########

while t<= max_load:
    u_Lx.t=t*ut
    u_old.assign(u_conv)
    c_old.assign(c_conv)
    solver_disp.solve()
    u_new.assign(u)
    solver_c.solve()
    u_conv.assign(u)
    c_conv.assign(p)
    t+=deltaT
    #print ("u_new: ",norm(u))
    #print ("u_old: ",norm(u_old))
    #print ("c_old: ",norm(c_old))
 
    
(ux,uy) = split(u)
plot(ux,title = 'u_dispx')
plt.show()
plot(uy,title = 'u_dispy')
plt.show()
plot(p,title = 'Phase field')
plt.show()

print ("solution done with no error")

######COMMENTS:
# 1. Mesh is refined: Line 11
# 2. l_0 is taken .011 instead of 7.5e-3.
# 3. No solution for u and c is obtained if crack is taken as reference axis:: Line 48
#  ** Warning: Found no facets matching domain for boundary condition. (with both u as 0)
#  Also for middle subdomain, I have taken: x[0]-0.5 :0.5 subtraction to let it know where crack starts.
#
# 4. History definition: Line 80
#    Instead of definition from Borden paper, alternate Definition is taken.	
#
# 5. Variational problem:
#    Definition of E_c according to paper Borden is not working properly (crack is propagated all over ).::Line 108
#    However alternate definition (from another Paper) is working fine and giving crack at middle point.
