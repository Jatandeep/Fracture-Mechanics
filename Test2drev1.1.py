from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
from dolfin import *
from ufl import nabla_div


mesh = UnitSquareMesh(8,8)

#### material parameters ######
c = 1e-3
k = 1e-6
Gc = 2.7
E = 210e3
nu = 0.3
l_0 = 7.5e-3

mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

#### Spaces #####
W = VectorFunctionSpace(mesh,'CG',1)
u , v = TrialFunction(W),TestFunction(W)
print (u)
V = FunctionSpace(mesh,'CG',1)
p , q = TrialFunction(V), TestFunction(V)


####### classes #########

class top(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]) < tol and on_boundary#!

class bottom(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]) < tol and on_boundary
'''
class middle(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]) < tol and x[0] < tol
        
def eps(v):
    return sym(grad(v))
    
def sigma(v):
    return lmbda*tr(eps(v))*Identity(2) + 2.0*mu*eps(v) '''
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    
def sigma(u):
    return lmbda*nabla_div(u)*Identity(2) + 2*mu*epsilon(u)
    
kb = lmbda*(1+nu)/3.0/nu

def hist(u):
    return 0.5*kb*( tr(eps(u)) - abs(tr(eps(u))) )**2.0 + ((1.0-k)*c**2.0 +k)*(0.5*kb*abs(tr(eps(u)))**2.0 + mu*tr(dev(eps(u))*dev(eps(u))) )
    
####### Boundary Conditions ########

Top = top()
Bottom = bottom()
#Middle = middle()

print ('hello 1')

print ('hello 2')
 
u_Lx = Expression("t",t = 0.0, degree=0)
ftx =  DirichletBC(W.sub(1),Constant(0.0),Top,method='pointwise')
fty =  DirichletBC(W.sub(0),u_Lx,Top,method='pointwise')

fbx =  DirichletBC(W.sub(0),Constant(0.0),Bottom,method='pointwise')
fby =  DirichletBC(W.sub(1),Constant(0.0),Bottom,method='pointwise')

bc_u = [ftx,fty,fbx,fby]	

#bc_c = DirichletBC(V,Constant(1.0),Middle,method='pointwise')

######## initialization ############

u_old, u_conv, u_new = Function(W),Function(W), Function(W)
c_old, c_conv = Function(V),Function(V)
print ('hello 3')
f = Constant((1.0,0))
L = dot(f, v)*dx 
####### Variational Problem ########
a=inner(grad(v),sigma(u))*dx - f*v*dx
#E_u=((1-k)*c**2.0 + k)*inner(grad(v),sigma(u))*dx

#E_c = (((4.0*l_0*(1.0-k)* hist(u_new))/Gc + 1.0)*inner(p,q) - (4.0*l_0**2.0*inner(grad(p),grad(q)) -1.0))*dx
print ('hello 4')
u = Function(W)
#p = Function(V)
print (u(0.5,0.5))
#a, L = lhs(E_u), rhs(E_u)
# Solve problem
solve(a == L, u, bc_u)
print(u(0.5,0.5))
'''
p_disp = LinearVariationalProblem(lhs(E_u),rhs(E_u),u,bc_u)
#p_c = LinearVariationalProblem(lhs(E_c),rhs(E_c),p, bc_c)

###### solver #######
solver_disp = LinearVariationalSolver(p_disp)
#solver_c = LinearVariationalSolver(p_c)
print ('hello 5')
t = 1e-5
#max_load = 0.03
max_load = 1e-5
deltaT  = 1e-5
ut = 1.0

###### Displacement controlled test ########
while t <= max_load:
	u_Lx.t=t*ut
	u_old.assign(u_conv)
	#c_old.assign(c_conv)
	iter = 1.0
	toll = 1e-2
	#err = 1e-2
	err = 1
	#maxiter = 100.0
	while err > toll:
		print ('hello 6')
		solver_disp.solve()
		print ('hello 7')
		#u_new.assign(u)
		#solver_c.solve()
		print (u(0.5,0.5))
		#print (p)
		print (u_old)
		#print (c_old)
		try:
			err = errornorm(u,u_old, norm_type = 'l2',degree_rise =0, mesh = None)/norm(u)
		except ZeroDivisionError:
			err = 2.0
		print ('hello 7.1')
		#try:
		#	err_c = errornorm(p,c_old, norm_type = 'l2',degree_rise = 0, mesh = None)/norm(p)
		#except ZeroDivisionError:
		#	err_c = 1.0
		print (err)
		#print (err_c)
		#err = max(err_u,err_c)
		u_old.assign(u)
		#c_old.assign(p)
		iter = iter + 1.0
		if err < toll:
			print("Solution Converge after:",iter)
			u_conv.assign(u)
			c_conv.assign(p)

			print ('Hello 8')
			(ux,uy) = split(u)
			print (ux)
			print (uy)
			print ('Hello 9')
			plot(ux,key = 'ux',title = 'u_dispx')
			plt.show()
			#plot(ux.function_space().mesh())
			print ('Hello 10')
			plot(uy,key ='uy',title = 'u_dispy')
			print ('Hello 11')
			#plot(p,range_min = 0.,range_max = 1.,key = 'c',title = 'c%.5f'%(t))
			print ('Hello 12')
	t+=deltaT
print ("solution done with no error :")
'''
plt.show()
#interactive()
