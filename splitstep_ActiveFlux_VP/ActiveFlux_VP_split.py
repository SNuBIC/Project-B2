
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import os
import time


class AF_Vlasov:
	
	def __init__(self, scheme='slice', splitting='Strang',init_cond='LD', nx=32, nv=32, capture_timeseries=True, snapshots=False):
		
		#parameters 
		self.init_cond =  init_cond
		self.scheme = scheme
		self.timesplitting = splitting
		self.snapshots = snapshots
		
		self.capture_timeseries = capture_timeseries
		
		self.nx, self.nv = nx, nv
		#self.B = 2
		self.sizex, self.sizev = 2*self.nx+1, 2*self.nv+1
		
		
		#dictionaries
		self.xs_dict =  {'LD' : np.linspace(-2.*np.pi, 2.*np.pi, self.sizex), 
					'TS' : np.linspace(-5.*np.pi, 5.*np.pi, self.sizex),
					'SLD' : np.linspace(-2.*np.pi, 2.*np.pi, self.sizex)
					}
		self.vs_dict =  {'LD' : np.linspace(-5., 5., self.sizev), 
					'TS' : np.linspace(-10., 10., self.sizev),
					'SLD' : np.linspace(-5., 5., self.sizev)
					}
					
		self.init_funcs_dict = {'LD' : self.init_LD, 
								'TS' : self.init_TS ,
								'SLD' : self.init_SLD}
		
		self.poisson_funcs_dict = {'GS' : self.solve_poisson_GS, 
									'FFT' : self.solve_poisson_FFT}
									
		self.calc_rho_funcs_dict = {'allDOF' : self.calc_rho_allDOF, 
									'cons' : self.calc_rho_consistent}
									
		self.calc_gradient_funcs_dict = {'FD' : self.gradient_phi, 'FFT' : self.gradient_phi_FFT}
		
		self.stepX_funcs_dict = {'slice' : self.stepX_slice, 
								'fluxintegral' : self.stepX_fluxintegral,
								'sliceDD' : self.stepX_sliceDD}
								
		self.stepV_funcs_dict = {'slice' : self.stepV_slice, 
								'fluxintegral' : self.stepV_fluxintegral, 
								'sliceDD' : self.stepV_sliceDD}
		
		#spatial
		self.xs = self.xs_dict[self.init_cond]
		self.vs = self.vs_dict[self.init_cond]
		
		self.dx = self.xs[2] - self.xs[0]
		self.dvmin = 0.01953125
		self.dv = self.vs[2] - self.vs[0]
		
		#plasma parameters
		self.me, self.mi = 1., 1.
		self.qe, self.qi = -1., 1.
		self.Te, self.Ti = 1., 1.
		
		#temporal
		tmaxs = {'LD' : 0.5, 'TS' : 0.5, 'SLD' : 0.5}
		self.t, self.tmax = 0., 0.5  #for convergence LD : 1, TS : 0.5
		#self.cfl = 0.5*0.63661977236758134308
		self.cfl = 0.63661977236758134308 #cfl = 2/pi
		#self.cfl = 1.
		
		if (self.cfl > 1.):
			raise ValueError('CFL must be \leq 1, otherwise unstable')
		
		elif (self.cfl > 0.5) and (self.scheme == 'sliceDD'): #<-- handle cfl-restriction DD
			raise ValueError('CFL for DD must be \leq 0.5, otherwise unstable ')
			
		
		self.dt = self.cfl * self.dx / np.max(self.vs)
		
		if self.timesplitting == 'Yoshida':
			self.gamma1 = 1./(2. - 2.**(1./3.))
			self.gamma2 = -2.**(1./3.)/(2. - 2.**(1./3.))
		
		
		#set poisson method for all schemes
		self.poisson_solver = 'FFT' #<--FFT or GS
		self.calc_gradient_method = 'FFT' #<-- FFT or FD
		
		#scheme-specific settings
		self.stepX_method = self.scheme
		self.stepV_method = self.scheme
		
		if (self.scheme == 'slice'):
			
			self.perform_init_averaging = True
			self.perform_init_averaging_analytic = False
			self.average_E = True #<-- average E (1D) after every step: o--x--o
			self.calc_rho_method = 'cons' 
			self.perform_center_reconstruction = True #<-- reconstruct after computing rho: o--o--o
			self.average_f = False #<-- average f (2D) after every step 
			
		elif (self.scheme == 'fluxintegral'):
			
			self.perform_init_averaging = True
			self.perform_init_averaging_analytic = False
			self.average_E = True #<-- average E (1D) after every step: o--x--o
			self.calc_rho_method = 'cons'
			self.perform_center_reconstruction = True #<-- reconstruct after computing rho: o--o--o
			self.average_f = False #<-- average f (2D) after every step 
			
		elif (self.scheme == 'sliceDD'):
			
			self.perform_init_averaging = False
			self.perform_init_averaging_analytic = False
			self.average_E = False
			self.calc_rho_method = 'allDOF'
			self.perform_center_reconstruction = False
			self.average_f = True  
		
		#setting methods that are used during self.solve()
		self.stepX = self.stepX_funcs_dict[self.stepX_method]
		self.stepV = self.stepV_funcs_dict[self.stepV_method]
		
		self.solve_poisson = self.poisson_funcs_dict[self.poisson_solver]
		self.calc_rho = self.calc_rho_funcs_dict[self.calc_rho_method]
		self.calc_gradient = self.calc_gradient_funcs_dict[self.calc_gradient_method]
		
		#init f
		self.fe, self.fi = self.init_funcs_dict[self.init_cond]()
		
		if self.perform_init_averaging:
			self.fe = self.init_averaging(self.fe)
			self.fi = self.init_averaging(self.fi)
		
		if self.perform_init_averaging_analytic:
			
			if (self.init_cond == 'LD'):
				self.fe = self.init_averaging_analytic_LD(self.fe, fe = True)
				self.fi = self.init_averaging_analytic_LD(self.fi, fe = False)
				
			elif (self.init_cond == 'TS'):
				self.fe = self.init_averaging_analytic_TS(self.fe, fe = True)
				self.fi = self.init_averaging_analytic_TS(self.fi, fe = False)
		
		#init other fields (to None or comppute at t=0)
		self.rho = None
		self.phi = None
		self.E = None
		self.E = np.zeros(self.sizex)
		self.update_electrostatic()
		
		#output setting
		self.print_info_timestep = True
		
		self.dtOutput = 1. * self.dt
		
		self.plot_time_series = False
		self.output_time_series = False
		
		if self.capture_timeseries:
			self.plot_time_series = True 
			self.output_time_series = True
		
		if self.plot_time_series:
			self.output_directory = f'output_{self.init_cond}_{self.scheme}_{self.timesplitting}_{self.nx}_{self.nv}'
			if self.output_directory in os.listdir():
				os.system(f'rm -r {self.output_directory}')
			os.system(f'mkdir {self.output_directory}')
			
			
			if self.snapshots:
				self.st = 0.
		
		if self.plot_time_series:
			self.ts = []
			self.Esqs = []
			self.Masses = []
			self.Momentums = []
			self.nEs = []
			self.Ekins = []
			self.Epots = []
			self.Etotals = []
			self.Entropys = []
			self.L1Norms = []
			self.L2Norms = []
			
			self.snapshotsarr = []
			self.snapshotspointsarr = []
		
		
		print('---------------------------------------')
		print('Set-Up AF-Vlasov-Poisson')
		print('settings : ', self.scheme, self.timesplitting, self.init_cond)
		print('(nx,nv) : ', (self.nx, self.nv))
		print('t : ', self.t, 'tmax :', self.tmax)
		print('cfl : ', self.cfl)
		print('dt :', self.dt)
		print(' ')
		
		
	def __str__(self):
		import inspect
		return inspect.getsource(self.__init__)
		
	def __del__(self):
		print(f'Deleted {self.nx}')
	
	#init
	def init_averaging(self, arr):
		
		arr_temp = np.zeros(arr.shape)
		
		#edges
		for i in range(0, self.sizex, 2):
			for j in range(0, self.sizev, 2):
				arr_temp[i, j] = arr[i, j]
			
		#vertical edges
		for i in range(0, self.sizex, 2):
			for j in range(1, self.sizev-1, 2):
				arr_temp[i, j] = (1./6.) * (arr[i,j-1] + 4.*arr[i,j] + arr[i,j+1])
					
		#horizontal edges
		for i in range(1, self.sizex-1, 2):
			for j in range(0, self.sizev, 2):
				arr_temp[i, j] = (1./6.) * (arr[i-1, j] + 4.*arr[i,j] + arr[i+1,j])
			
		#centers 
		for i in range(1, self.sizex-1, 2):
			for j in range(1, self.sizev-1, 2):
				sum_nodes = arr[i-1,j-1] + arr[i-1,j+1] + arr[i+1,j-1] + arr[i+1,j+1]
				sum_edges = arr[i-1,j] + arr[i+1,j] + arr[i,j-1] + arr[i,j+1]
				center = arr[i,j]
					
				arr_temp[i, j] = (1./36.) * (sum_nodes + 4.*sum_edges + 16.*center)
		
		return arr_temp
	
	def init_averaging_analytic_LD(self, arr, fe = True):
		
		from scipy.special import erf as scerf
		
		arr_temp = np.zeros(arr.shape)
		
		eps = 0.001
		k = 0.5
		
		#nodes
		for i in range(0, self.sizex, 2):
				for j in range(0, self.sizev, 2):
					arr_temp[i, j] = arr[i, j]
		
		#x-averages
		A = (1./(np.sqrt(2.*np.pi)*k*self.dx))
		for i in range(1, self.sizex-1, 2):
			x = self.xs[i]
			xp = x + 0.5*self.dx 
			xm = x - 0.5*self.dx
			for j in range(0, self.sizev, 2):
				v = self.vs[j]
				
				if fe:
					arr_temp[i, j] = A*np.exp(-0.5*v**2)*((np.sin(k*xp)-np.sin(k*xm))*eps + k*(xp-xm))
				else:
					arr_temp[i, j] = (A*k)*np.exp(-0.5*v**2)*(xp-xm)
					
		#v-averages
		A = (1./(2.*self.dv))
		B = (1./np.sqrt(2))
			
		for j in range(1, self.sizev-1, 2):
			v = self.vs[j]
			vp = v + 0.5*self.dv
			vm = v - 0.5*self.dv
			for i in range(0, self.sizex, 2):
				x = self.xs[i]
				
				if fe:
					arr_temp[i, j] = A*(scerf(B*vp)-scerf(B*vm))*(np.cos(k*x)*eps + 1.)
				else:
					arr_temp[i, j] = A*(scerf(B*vp)- scerf(B*vm))
		
		#cell-averages
		A = (1./(2.*k*self.dx*self.dv))
		B = (1./np.sqrt(2))
			
		for i in range(1, self.sizex-1,2):
			x = self.xs[i]
			xp = x + 0.5*self.dx 
			xm = x - 0.5*self.dx
			for j in range(1, self.sizev-1, 2):
				v = self.vs[j]
				vp = v + 0.5*self.dv
				vm = v - 0.5*self.dv
				
				if fe:
					arr_temp[i, j] = A * (scerf(B*vp)-scerf(B*vm)) * ((np.sin(k*xp)-np.sin(k*xm))*eps + k*(xp-xm))
				else:
					arr_temp[i, j] = (A*k)*(scerf(B*vp)-scerf(B*vm))*(xp-xm)
		
		return arr_temp
	
	def init_averaging_analytic_TS(self, arr, fe=True):
		
		from scipy.special import erf as scerf
		
		arr_temp = np.zeros(arr.shape)
		
		k = 0.2
		eps = 0.001
		v0 = 3. #1.3, 2.4, 3.0
		
		#nodes
		for i in range(0, self.sizex, 2):
				for j in range(0, self.sizev, 2):
					arr_temp[i, j] = arr[i, j]
		
		#x-averages
		A = (1./(2.**(3./2.) *np.sqrt(np.pi)*k*self.dx))
		for i in range(1, self.sizex-1, 2):
			x = self.xs[i]
			xp = x + 0.5*self.dx 
			xm = x - 0.5*self.dx
			for j in range(0, self.sizev, 2):
				v = self.vs[j]
				
				if fe:
					arr_temp[i, j] = A*(np.exp(-0.5*(v-v0)**2)+np.exp(-0.5*(v+v0)**2))*((np.sin(k*xp)-np.sin(k*xm))*eps+k*(xp-xm))
				else:
					arr_temp[i, j] = (A*k)*(np.exp(-0.5*(v-v0)**2)+np.exp(-0.5*(v+v0)**2))*(xp-xm)
		
		#v-averages
		A = (1./(4.*self.dv))
		B = (1./np.sqrt(2))
		C = 0.5*np.sqrt(2)
		
		for j in range(1, self.sizev-1, 2):
			v = self.vs[j]
			vp = v + 0.5*self.dv
			vm = v - 0.5*self.dv
			for i in range(0, self.sizex, 2):
				x = self.xs[i]
				
				if fe:
					arr_temp[i, j] = A*(scerf(C*(vp+v0))+scerf(C*(vp-v0))-scerf(C*(vm+v0))-scerf(C*(vm-v0)))*(np.cos(k*x)*eps + 1.)
				else:
					arr_temp[i, j] = A*(scerf(C*(vp+v0))+scerf(C*(vp-v0))-scerf(C*(vm+v0))-scerf(C*(vm-v0)))
		
		#cell-averages
		A = (1./(4.*k*self.dx*self.dv))
		B = 0.5*np.sqrt(2)
			
		for i in range(1, self.sizex-1,2):
			x = self.xs[i]
			xp = x + 0.5*self.dx 
			xm = x - 0.5*self.dx
			for j in range(1, self.sizev-1, 2):
				v = self.vs[j]
				vp = v + 0.5*self.dv
				vm = v - 0.5*self.dv
				
				if fe:
					arr_temp[i, j] = A*(scerf(B*(vp+v0))+scerf(B*(vp-v0))-scerf(B*(vm+v0))-scerf(B*(vm-v0)))*((np.sin(k*xp)-np.sin(k*xm))*eps+k*(xp-xm))
				else:
					arr_temp[i, j] = (A*k)*(scerf(B*(vp+v0))+scerf(B*(vp-v0))-scerf(B*(vm+v0))-scerf(B*(vm-v0)))*(xp-xm)

		
		
		return arr_temp
	
	def averages_to_points(self, arr):
		
		arr_temp = np.zeros(arr.shape)
		arr_temp[:, :] = arr[:, :]
		
		#x-averages
		for i in range(1, self.sizex-1, 2):
			for j in range(0, self.sizev, 2):
				arr_temp[i, j] = 0.25 * (6.*arr[i, j] - arr[i-1, j] - arr[i+1, j])
				
		#v-averages
		for j in range(1, self.sizev-1, 2):
			for i in range(0, self.sizex, 2):
				arr_temp[i, j] = 0.25 * (6.*arr[i, j] - arr[i, j-1] - arr[i, j+1])
		
		#cell averages
		for i in range(1, self.sizex-1, 2):
			for j in range(1, self.sizev-1, 2):
				sum_nodes = arr[i-1,j-1] + arr[i+1,j-1] + arr[i-1,j+1] + arr[i+1,j+1]
				sum_edges = arr_temp[i-1,j] + arr_temp[i+1,j] + arr_temp[i,j+1] + arr_temp[i,j-1]
				
				arr_temp[i, j] = (1./16.) * (36.*arr[i, j] - sum_nodes - 4.*sum_edges)
		
		return arr_temp
		
	def init_LD(self):
		
		fe = np.zeros((self.sizex, self.sizev))
		fi = np.zeros((self.sizex, self.sizev))
		
		eps = 0.001
		k = 0.5
		for i in range(self.sizex):
			x = self.xs[i]
			for j in range(self.sizev):
				v = self.vs[j]
				
				fe[i, j] = np.sqrt(1./(2.*np.pi)) * np.exp(-0.5*v**2) * (1. + eps * np.cos(k*x)) 
				fi[i, j] = np.sqrt(1./(2.*np.pi)) * np.exp(-0.5*v**2) * (1. + 0.0 * np.cos(k*x))
					
		return fe, fi
		
	def init_SLD(self):
		
		fe = np.zeros((self.sizex, self.sizev))
		fi = np.zeros((self.sizex, self.sizev))
		
		eps = 0.5
		k = 0.5
		for i in range(self.sizex):
			x = self.xs[i]
			for j in range(self.sizev):
				v = self.vs[j]
				
				fe[i, j] = np.sqrt(1./(2.*np.pi)) * np.exp(-0.5*v**2) * (1. + eps * np.cos(k*x)) 
				fi[i, j] = np.sqrt(1./(2.*np.pi)) * np.exp(-0.5*v**2) * (1. + 0.0 * np.cos(k*x))
		
		return fe, fi
		
	def init_TS(self):
		
		fe = np.zeros((self.sizex, self.sizev))
		fi = np.zeros((self.sizex, self.sizev))
		
		k = 0.2
		eps = 0.001
		v0 = 3. #1.3, 2.4, 3.0
		
		for i in range(self.sizex):
			x = self.xs[i]
			for j in range(self.sizev):
				v = self.vs[j]
				
				fe[i, j] = 0.5 * np.sqrt(1./(2.*np.pi)) * (np.exp(-0.5*(v-v0)**2 ) + np.exp(-0.5*(v+v0)**2)) \
				* (1. + eps * np.cos(k*x))
				
				fi[i, j] = 0.5 * np.sqrt(1./(2.*np.pi)) * (np.exp(-0.5*(v-v0)**2 ) + np.exp(-0.5*(v+v0)**2)) \
				* (1. + 0. * np.cos(k*x))
		
		return fe, fi
	
	def reassemble1D(self, arr):
		size = np.shape(arr)[0]
		arr_temp = np.zeros(size+1) #<-- include periodic point again
		
		arr_temp[:-1] = arr[:]
		arr_temp[-1] = arr[0]
		return arr_temp
		
	def reassemble2D(self, arr):
		sizex, sizev = np.shape(arr)
		arr_temp = np.zeros((sizex+1, sizev+1))
		
		arr_temp[:-1, :-1] = arr[:, :]
		arr_temp[-1, :-1] = arr[0, :]
		arr_temp[:-1, -1] = arr[:, 0]
		arr_temp[-1, -1] = arr[0, 0]
		return arr_temp
		
	#solve advection
	def advection_AF(self, arr, speed, dt, dx, conservation = True):
		
		size = np.shape(arr)[0]
		
		nu =  speed * dt / dx
		arr_temp = np.zeros(arr.shape)
		
		if (nu >= 0.):
			
			nu = np.abs(nu)
			
			#interface update formula
			arr_temp[0::2] = (nu*(3.*nu - 2.)*np.roll(arr, 2) 
			+ 6.*nu*(1.-nu) * np.roll(arr,1)
			+ (1.-nu)*(1.-3.*nu) * arr)[0::2]
			
			#average update formula
			if conservation:
				arr_temp[1::2] = (nu**2 * (nu -1.)*np.roll(arr, 3) 
				+ nu**2 *(3.-2.*nu)*np.roll(arr,2) 
				+ nu*(1.-nu) * np.roll(arr,1) 
				+ (1.-nu)**2 * (1.+2.*nu) * arr 
				- nu*(1.-nu)**2 * np.roll(arr, -1))[1::2]
			
			else:
				arr_temp[1::2] = arr[1::2]
			
		else:
			
			nu = np.abs(nu)
			
			arr_temp[0::2] = ((1.-nu) * (1.-3.*nu) * arr
			+ 6.*nu*(1.-nu) * np.roll(arr,-1)
			+ nu*(3.*nu - 2.) * np.roll(arr,-2))[0::2]
			
			if conservation:
				arr_temp[1::2] = (nu**2 * (nu-1.) * np.roll(arr, -3)
				+ nu**2 * (3.-2.*nu) * np.roll(arr, -2) 
				+ nu * (1.-nu) * np.roll(arr,-1)
				+ (1.-nu)**2 * (1.+2.*nu) * arr
				- nu*(1.-nu)**2 * np.roll(arr, 1))[1::2]
				
			else:
				arr_temp[1::2] = arr[1::2]
		
		return arr_temp
		
	def advection_AFDD(self, arr, speed, dt, dx):
		
		size = np.shape(arr)[0]
		
		a = 1. #<-- for third order + stability a,b should satisfy condition: (2./3.)*a+(1./6.)*b+(1./6.)*b=1.
		b = 1.
		
		#a = 0. #this yields second order scheme 
		#b = 0.
		
		nu = 2.* speed * dt / dx
		arr_temp = np.zeros(arr.shape)
		
		if (nu >= 0.):
			
			nu = np.abs(nu)
			
			#interface
			arr_temp[0::2] = (
			b*nu*(nu-2.)*(2.*nu-1.)/48. * np.roll(arr, 4)
			+ -b*nu*(nu**2 - 4.*nu+2.)/12. * np.roll(arr, 3)
			+ nu*(2.*b*nu**2 -23.*b*nu +14.*b+24.*nu-24.)/48. * np.roll(arr, 2)
			+ nu*(3.*b*nu-2.*b-6.*nu+12.)/6. * np.roll(arr, 1)
			+ -(2.*b*nu**3 + 19.*b*nu**2 -14.*b*nu-24.*nu**2 +72.*nu-48.)/48. * arr
			+ b*nu*(nu**2 + 2.*nu-2.)/12. * np.roll(arr, -1)
			+ -b*nu*(2.*nu**2 + nu -2.)/48. * np.roll(arr, -2)
			)[0::2]
			 
			#center 
			arr_temp[1::2] = (
			(a*nu**3 / 6. + a*nu**2 / 3. - a*nu/3. - nu**2 + 1.) * arr
			+ (-a*nu**3 / 12. - a*nu**2 / 24. + a*nu/12. + 0.5*nu**2 - 0.5*nu) * np.roll(arr, -1)
			+ (-3.*a*nu**2 / 4. + 0.5*a*nu + 0.5*nu**2 + 0.5*nu) * np.roll(arr, 1)
			+ (-a*nu**3 / 6. + 2.*a*nu**2 / 3. - a*nu/3.) * np.roll(arr, 2)
			+ (a*nu**3 / 12. - 5.*a*nu**2 / 24. + a*nu/12.) * np.roll(arr, 3)
			)[1::2]
			
		else:
			
			nu = np.abs(nu)
			
			#interface
			arr_temp[0::2] = (
			b*nu*(nu-2.)*(2.*nu-1.)/48. * np.roll(arr, -4)
			+ -b*nu*(nu**2 - 4.*nu+2.)/12. * np.roll(arr, -3)
			+ nu*(2.*b*nu**2 -23.*b*nu +14.*b+24.*nu-24.)/48. * np.roll(arr, -2)
			+ nu*(3.*b*nu-2.*b-6.*nu+12.)/6. * np.roll(arr, -1)
			+ -(2.*b*nu**3 + 19.*b*nu**2 -14.*b*nu-24.*nu**2 +72.*nu-48.)/48. * arr
			+ b*nu*(nu**2 + 2.*nu-2.)/12. * np.roll(arr, 1)
			+ -b*nu*(2.*nu**2 + nu -2.)/48. * np.roll(arr, 2)
			)[0::2]
			
			#center
			arr_temp[1::2] = (
			(a*nu**3 / 6. + a*nu**2 / 3. - a*nu/3. - nu**2 + 1.) * arr
			+ (-a*nu**3 / 12. - a*nu**2 / 24. + a*nu/12. + 0.5*nu**2 - 0.5*nu) * np.roll(arr, 1)
			+ (-3.*a*nu**2 / 4. + 0.5*a*nu + 0.5*nu**2 + 0.5*nu) * np.roll(arr, -1)
			+ (-a*nu**3 / 6. + 2.*a*nu**2 / 3. - a*nu/3.) * np.roll(arr, -2)
			+ (a*nu**3 / 12. - 5.*a*nu**2 / 24. + a*nu/12.) * np.roll(arr, -3)
			)[1::2]
			
		return arr_temp
	
	def stepX_slice(self, arr, dt):
		
		arr_temp = arr.copy()
		arr_slice = np.zeros(self.sizex-1)
		
		for j in range(0, self.sizev-1):
			
			speed = self.vs[j]
			
			arr_slice[:] = arr_temp[:, j]
			arr_slice = self.advection_AF(arr_slice, speed, dt, self.dx)
			arr_temp[:, j] = arr_slice[:]
			
		return arr_temp
	
	def stepV_slice(self, arr, dt, q, m):
		
		arr_temp = arr.copy()
		arr_slice = np.zeros(self.sizev-1)
		
		for i in range(0, self.sizex-1):
			
			speed = (q/m) * self.E[i]
			
			arr_slice[:] = arr_temp[i, :]
			arr_slice = self.advection_AF(arr_slice, speed, dt, self.dv)
			arr_temp[i, :] = arr_slice[:]
			
		return arr_temp 
	
	def stepX_sliceDD(self, arr, dt):
		
		arr_temp = arr.copy()
		arr_slice = np.zeros(self.sizex-1)
		
		for j in range(0, self.sizev-1):
			speed = self.vs[j]
			
			arr_slice[:] = arr_temp[:, j]
			arr_slice = self.advection_AFDD(arr_slice, speed, dt, self.dx)
			arr_temp[:, j] = arr_slice[:]
			
		return arr_temp
	
	def stepV_sliceDD(self, arr, dt, q, m):
		
		arr_temp = arr.copy()
		arr_slice = np.zeros(self.sizev-1)
		
		for i in range(0, self.sizex-1):
			
			speed = (q/m) * self.E[i]
			
			arr_slice[:] = arr_temp[i, :]
			arr_slice = self.advection_AFDD(arr_slice, speed, dt, self.dv)
			arr_temp[i, :] = arr_slice[:]
			
		return arr_temp
	
	def stepX_fluxintegral(self, arr, dt):
		
		arr_temp = arr.copy() #<- f^n+1
		arrold = arr.copy() #<- f^n
		arrhalf = arr.copy() #<- f^n+1/2
		
		arr_slice = np.zeros(self.sizex-1)
		
		#interfaces/vertical edges --> normal AF setup (o--x--o)
		for j in range(0, self.sizev -1 , 2):
			
			speed = self.vs[j]
			
			arr_slice[:] = arrhalf[:, j]
			arr_slice = self.advection_AF(arr_slice, speed, 0.5*dt, self.dx, False)
			arrhalf[:, j] = arr_slice[:]
			
			arr_slice[:] = arr_temp[:, j]
			arr_slice = self.advection_AF(arr_slice, speed, 1.*dt, self.dx, True)
			arr_temp[:, j] = arr_slice[:]
		
		for j in range(1, self.sizev-1, 2):
			
			speed = self.vs[j]
			
			arr_slice[:] = arrhalf[:, j]
			arr_slice = self.advection_AF(arr_slice, speed, 0.5*dt, self.dx, False)
			arrhalf[:, j] = arr_slice[:]
			
			arr_slice[:] = arr_temp[:, j]
			arr_slice = self.advection_AF(arr_slice, speed, 1.*dt, self.dx, False)
			arr_temp[:, j] = arr_slice[:]
			
		
		#conservation update --> 9 point integral
		arr_temp = self.reassemble2D(arr_temp)
		arrhalf = self.reassemble2D(arrhalf)
		arrold = self.reassemble2D(arrold)
		
		frold = np.zeros(self.sizex)
		frhalf = np.zeros(self.sizex)
		fr = np.zeros(self.sizex)
		
		for j in range(1, self.sizev-1, 2):
			
			frold [0:self.sizex:2] = 0.25*(6.*arrold[0:self.sizex:2,j]-arrold[0:self.sizex:2,j+1]-arrold[0:self.sizex:2,j-1])
			frhalf[0:self.sizex:2] = 0.25*(6.*arrhalf[0:self.sizex:2,j]-arrhalf[0:self.sizex:2,j+1]-arrhalf[0:self.sizex:2,j-1])
			fr    [0:self.sizex:2] = 0.25*(6.*arr_temp[0:self.sizex:2,j]-arr_temp[0:self.sizex:2,j+1]-arr_temp[0:self.sizex:2,j-1])
			
			
			for i in range(1, self.sizex-1, 2):
				
				flux_left = (1./36.) * (
				self.vs[j-1]*(arrold[i-1,j-1]+4.*arrhalf[i-1,j-1]+ arr_temp[i-1,j-1])
				+ self.vs[j+1]*(arrold[i-1,j+1]+4.*arrhalf[i-1,j+1]+ arr_temp[i-1,j+1])
				+ 4.*self.vs[j]*(frold[i-1]+4.*frhalf[i-1]+fr[i-1]) 
				)
				
				flux_right = (1./36.) * (
				self.vs[j-1]*(arrold[i+1,j-1]+4.*arrhalf[i+1,j-1]+ arr_temp[i+1,j-1])
				+ self.vs[j+1]*(arrold[i+1,j+1]+4.*arrhalf[i+1,j+1]+ arr_temp[i+1,j+1])
				+ 4.*self.vs[j]*(frold[i+1]+4.*frhalf[i+1]+fr[i+1]) 
				)
				
				arr_temp[i, j] = arr_temp[i, j] - (dt/self.dx) * (flux_right - flux_left)
		
		arr_temp = arr_temp[:-1, :-1]
		return arr_temp
		
	def stepV_fluxintegral(self, arr, dt, q, m):
		
		arr_temp = arr.copy() #<- f^n+1
		arrold = arr.copy() #<- f^n
		arrhalf = arr.copy() #<- f^n+1/2
		
		arr_slice = np.zeros(self.sizev-1)
		
		#interfaces/vertical edges --> normal AF setup (o--x--o)
		for i in range(0, self.sizex -1 , 2):
			
			speed = (q/m) * self.E[i]
			
			arr_slice[:] = arrhalf[i, :]
			arr_slice = self.advection_AF(arr_slice, speed, 0.5*dt, self.dv, False)
			arrhalf[i, :] = arr_slice[:]
			
			arr_slice[:] = arr_temp[i, :]
			arr_slice = self.advection_AF(arr_slice, speed, 1.*dt, self.dv, True)
			arr_temp[i, :] = arr_slice[:]
		
		for i in range(1, self.sizex -1 , 2):
			
			speed = (q/m) * self.E[i]
			
			arr_slice[:] = arrhalf[i, :]
			arr_slice = self.advection_AF(arr_slice, speed, 0.5*dt, self.dv, False)
			arrhalf[i, :] = arr_slice[:]
			
			arr_slice[:] = arr_temp[i, :]
			arr_slice = self.advection_AF(arr_slice, speed, 1.*dt, self.dv, False)
			arr_temp[i, :] = arr_slice[:]
			
		#reconstrut E o--x--o --> o--o--o
		Er = self.reassemble1D(self.E)
		for i in range(1, self.sizex-1, 2):
			Er[i] = 0.25 * (6.*Er[i] - Er[i-1] - Er[i+1])
		
		#conservation update --> 9 point integral
		arr_temp = self.reassemble2D(arr_temp)
		arrhalf = self.reassemble2D(arrhalf)
		arrold = self.reassemble2D(arrold)
		
		frold = np.zeros(self.sizev)
		frhalf = np.zeros(self.sizev)
		fr = np.zeros(self.sizev)
		
		for i in range(1, self.sizex-1, 2):
			
			frold[0:self.sizev:2] = 0.25*(6.*arrold[i,0:self.sizev:2]-arrold[i+1,0:self.sizev:2]-arrold[i-1,0:self.sizev:2])
			frhalf[0:self.sizev:2] = 0.25*(6.*arrhalf[i,0:self.sizev:2]-arrhalf[i+1,0:self.sizev:2]-arrhalf[i-1,0:self.sizev:2])
			fr[0:self.sizev:2] = 0.25*(6.*arr_temp[i,0:self.sizev:2]-arr_temp[i+1,0:self.sizev:2]-arr_temp[i-1,0:self.sizev:2])
			
			for j in range(1, self.sizev-1, 2):
				
				flux_bottom = (q/m) * (1./36.) * (
				Er[i-1] * (arrold[i-1,j-1]+4.*arrhalf[i-1,j-1]+arr_temp[i-1,j-1])
				+ Er[i+1] * (arrold[i+1,j-1] + 4.*arrhalf[i+1,j-1] + arr_temp[i+1,j-1])
				+ 4.*Er[i] * (frold[j-1] + 4.*frhalf[j-1] + fr[j-1])
				)
				
				flux_top = (q/m) * (1./36.) * (
				Er[i-1] * (arrold[i-1,j+1]+4.*arrhalf[i-1,j+1]+arr_temp[i-1,j+1])
				+ Er[i+1] * (arrold[i+1,j+1] + 4.*arrhalf[i+1,j+1] + arr_temp[i+1,j+1])
				+ 4.*Er[i] * (frold[j+1] + 4.*frhalf[j+1] + fr[j+1])
				)
				
				arr_temp[i, j] = arr_temp[i, j] - (dt/self.dv) * (flux_top - flux_bottom)
				
		
		arr_temp = arr_temp[:-1, :-1]
		return arr_temp
	
	#solve poisson
	def calc_rho_consistent(self):
		rho = np.zeros(self.sizex-1)
		
		for i in range(0, self.sizex-1):
			rho[i] = self.qe * self.dv * np.sum(self.fe[i, 1::2]) + self.qi * self.dv * np.sum(self.fi[i, 1::2]) 
			
		if self.perform_center_reconstruction:
			#evaluate reconstruction polynomial at center location 
			rho[1::2] = 0.25*(6.*rho[1::2] - np.roll(rho, 1)[1::2] - np.roll(rho,-1)[1::2])
			
		return rho
		
	def calc_rho_allDOF(self):
		rho = np.zeros(self.sizex-1)
		
		def simpson(arr, dx):
			return (dx/6.) * np.sum((np.roll(arr, -1) + np.roll(arr, 1) + 4.*arr)[1::2])
		
		for i in range(0, self.sizex-1):
			rho[i] = (self.qe*simpson(self.fe[i, :], self.dv) + self.qi*simpson(self.fi[i, :], self.dv))
		
		return rho 
		
		
	def solve_poisson_FFT(self):
		
		dx_val = 0.5*self.dx
		
		k = np.fft.fftfreq(self.sizex-1, d = dx_val) * 2. * np.pi #/ 4.*np.pi
		k[0] = 1.
		
		#print(k)
		
		rho_hat = fft(self.rho)
		phi_hat = rho_hat / (-k**2)
		phi_hat[0] = 0.
		phi_hat[int((self.sizex-1) / 2)] = 0.
		phi = np.real(ifft(phi_hat))
		
		phi -= np.mean(phi)
		
		return phi
	
	def solve_poisson_GS(self, order = 4):
		max_iter = int(5e5)
		
		dx_val = 0.5*self.dx
		phi = np.zeros(self.rho.shape)
		if self.phi is not None:
			phi[:] = self.phi[:] #<--initial guess
		
		it = 0
		err = 1e6
		tol = 1e-9
		while (err > tol):
			phiold = phi.copy()
			
			if order == 2:
				phi = 0.5*(np.roll(phi, -1) + np.roll(phi, 1) - dx_val**2 * self.rho)
			
			elif order == 4:
				phi = 0.5*(np.roll(phi,1)+np.roll(phi,-1)) - (dx_val**2 /24.)*(np.roll(self.rho,1)+np.roll(self.rho,-1)+10.*self.rho)
			
			phi -= np.mean(phi) 
			
			err = np.max(np.abs(phiold - phi))
			
			it += 1
			if (it > max_iter):
				print('max iter poisson GS')
				break
		
		#print('iter : ', it )
		#print('err_GS : ', err )
		return phi
	
	def gradient_phi(self, order = 2):
		
		dx_val = 0.5*self.dx
		
		if (order == 2):
			E = (np.roll(self.phi, -1) - np.roll(self.phi, 1))/(2.*dx_val)
		
		elif (order == 4):
			E = (-np.roll(self.phi,-2) + 8.*np.roll(self.phi,-1) - 8.*np.roll(self.phi,1) + np.roll(self.phi,2))/(12.*dx_val)
		
		return E
		
	def gradient_phi_FFT(self):
		dx_val = 0.5*self.dx
		
		k = np.fft.fftfreq(self.sizex-1, d = dx_val) * 2. *np.pi
		k[0] = 1.
		
		phi_hat = fft(self.phi)
		E_hat = 1j * k * phi_hat
		E_hat[0] = 0.
		E_hat[int((self.sizex-1)/2)] = 0.
		E = np.real(ifft(E_hat))
		
		E -= np.mean(E)
		
		return E
		
	def average_simpson(self, arr):
		size = arr.shape[0]
		
		arr[1::2] = (1./6.) * (np.roll(arr, -1)[1::2] + np.roll(arr, 1)[1::2] + 4.*arr[1::2])
		return arr
	
	
	#splittings
	def update_electrostatic(self):
		self.rho = self.calc_rho()
		self.phi = self.solve_poisson()
		self.E = self.calc_gradient()
		
	
	def update_lie_splitting(self, dt):
		
		#stepX
		self.fe = self.stepX(self.fe, dt)
				
		#stepV
		self.update_electrostatic()
		
		if self.average_E:
			self.E = self.average_simpson(self.E)
				
		self.fe = self.stepV(self.fe, dt, self.qe, self.me)
	
	def update_strang_splitting(self, dt):
		
		#stepX
		self.fe = self.stepX(self.fe, 0.5 * dt)
				
		#stepV
		self.update_electrostatic()
		
		if self.average_E:
			self.E = self.average_simpson(self.E)
				
		self.fe = self.stepV(self.fe, dt, self.qe, self.me)
				
		#stepX
		self.fe = self.stepX(self.fe, 0.5 * dt)
		
	def update_yoshida_splitting(self, dt):
		
		self.update_strang_splitting(dt = self.gamma1 * self.dt)
		self.update_strang_splitting(dt = self.gamma2 * self.dt)
		self.update_strang_splitting(dt = self.gamma1 * self.dt)
		
		
	#main loop
	def solve(self):
		
		#extract non-periodic parts 
		self.fe = self.fe[:-1, :-1] 
		self.fi = self.fi[:-1, :-1] 
		
		while (self.t < self.tmax - 0.5*self.dt):
			
			if self.print_info_timestep:
				if (np.mod(self.t, self.dtOutput) < self.dt):
					print('---------------------------------------')
					print('settings : ', self.scheme, self.timesplitting, self.init_cond)
					print('(nx,nv) : ', (self.nx, self.nv))
					print('t : ', self.t, 'tmax :', self.tmax)
					print('fe_max : ', np.max(self.fe))
					print('E_max :', np.max(self.E))
					print(' ')
				
			if self.plot_time_series:
				if (np.mod(self.t, self.dtOutput) < self.dt):
					fepoints = self.fe.copy()
					fipoints = self.fi.copy()
					Epoints = self.E.copy()
					
					self.fe = self.reassemble2D(self.fe) #<-- fe again in the shape (sizex, sizev) 
					self.fi = self.reassemble2D(self.fi)
					
					if self.average_f: #discrepancy
						self.fe = self.init_averaging(self.fe)
						self.fi = self.init_averaging(self.fi)
						self.E = self.average_simpson(self.E)
					
					if self.snapshots:
						if (np.abs(self.t-self.st) < 0.8*self.dt):
							
							print(f'{self.t} saved snapshot')
							
							self.snapshotsarr.append((self.st, self.fe[1::2, 1::2]))
							
							if self.average_f:
								self.snapshotspointsarr.append((self.st, self.reassemble2D(fepoints)))
							else:
								self.snapshotspointsarr.append((self.st, self.averages_to_points(self.fe)))
							
							self.st += 5.
						
					self.ts.append(self.t)
					self.Esqs.append(self.compute_Esq())
					
					self.Masses.append(self.compute_Mass())
					self.nEs.append(self.compute_nE())
					self.Momentums.append(self.compute_Momentum())
					self.Entropys.append(self.compute_Entropy())
					
					self.L1Norms.append(self.compute_LpNorm(p=1.))
					self.L2Norms.append(self.compute_LpNorm(p=2.))
					
					Ekin, Epot, Etotal = self.compute_Energy()
					
					self.Ekins.append(Ekin)
					self.Epots.append(Epot)
					self.Etotals.append(Etotal)
					
					self.fe = fepoints.copy()
					self.fi = fipoints.copy()
					self.E = Epoints.copy()
			
			if (self.timesplitting == 'Lie'):
				self.update_lie_splitting(dt=self.dt)
			
			elif (self.timesplitting == 'Strang'):
				self.update_strang_splitting(dt=self.dt)
			
			elif (self.timesplitting == 'Yoshida'):
				self.update_yoshida_splitting(dt=self.dt)
			
			self.t += self.dt
			
		
		#after main loop
		self.fe = self.reassemble2D(self.fe) #<-- fe again in the shape (sizex, sizev) 
		self.fi = self.reassemble2D(self.fi)
		
		self.rho = self.reassemble1D(self.rho)
		self.phi = self.reassemble1D(self.phi)
		self.E = self.reassemble1D(self.E)
		
		if self.average_f:
			self.fe = self.init_averaging(self.fe)
			self.fi = self.init_averaging(self.fi)
		
		if self.plot_time_series:
			if self.output_time_series:
				self.output_data()
			
			self.plot_Esq()
			self.plot_conservation()
		
	
	
	#diagnostic
	def compute_LpNorm(self, p=1.):
		Norm = (self.dx * self.dv * np.sum(np.abs(self.fe[1::2, 1::2])**p))**(1./p)
		return Norm
	
	def compute_Esq(self):
		return np.sqrt(np.sum(np.abs(self.E[1::2])**2))
	
	def compute_Mass(self):
		Mass = self.dx * self.dv * (np.sum(self.fe[1::2, 1::2]))
		return Mass
		
	def compute_nE(self):
		n = np.zeros(self.sizex-1)
		
		for i in range(self.sizex-1):
			n[i] =  self.dv * np.sum(self.fe[i, 1::2])
			
		nE = np.sum((n * self.E)[1::2])
		return nE
	
	def compute_Momentum(self):
		Momentum = 0.
		for i in range(self.sizex-1):
			Momentum += self.dx * self.dv * np.sum(self.fe[i, 1::2] * self.vs[1::2])
		return Momentum
		
	def compute_Energy(self):
		Ekin = self.dx * self.dv * np.sum(self.fe[1::2, 1::2] * (self.vs[1::2])**2)
		Epot = self.dx * np.sum(self.E[1::2]**2)
		return Ekin, Epot, Ekin+Epot
	
	def compute_Entropy(self):
		Entropy = self.dx * self.dv * np.sum(self.fe[1::2, 1::2] * np.log(self.fe[1::2, 1::2]))
		return Entropy
	
	#plotting
	def plot_Esq(self, show = False):
		plt.plot(self.ts, self.Esqs, linestyle = '--')
		
		if self.init_cond == 'LD':
			gamma = 0.1533
			plt.plot(self.ts, self.Esqs[0] * np.exp(-1.*gamma * np.array(self.ts)))
		
		plt.grid()
		plt.yscale('log')
		
		plt.savefig(f'{self.output_directory}/esqrd.png')
		
		if show:
			plt.show()
			
		plt.close()
		
	def plot_conservation(self, show = False):
		
		fig, axs = plt.subplots(7, figsize = (10, 10))
		
		axs[0].plot(self.ts, np.abs(self.Masses-self.Masses[0])/self.Masses[0])
		axs[0].set_yscale('log')
		axs[0].set_ylabel('relative Mass')
		
		axs[1].plot(self.ts, np.abs(self.nEs - self.nEs[0])/1.)
		axs[1].set_yscale('log')
		axs[1].set_ylabel('Delta nE')
		
		axs[2].plot(self.ts, np.abs(self.Momentums-self.Momentums[0]))
		axs[2].set_yscale('log')
		axs[2].set_ylabel('Momentum')
		
		axs[3].plot(self.ts, np.abs(self.Etotals - self.Etotals[0])/self.Etotals[0], label = 'Etot')
		axs[3].legend()
		axs[3].set_yscale('log')
		axs[3].set_ylabel('Energy')
		
		axs[4].plot(self.ts, self.Entropys, label = 'Entropy')
		#axs[4].legend()
		axs[4].set_yscale('log')
		axs[4].set_ylabel('Entropy')
		
		axs[5].plot(self.ts, (self.L1Norms-self.L1Norms[0])/self.L1Norms[0])
		axs[5].set_ylabel('rel. L1-Norm')
		axs[5].set_yscale('log')
		
		axs[6].plot(self.ts, (self.L2Norms-self.L2Norms[0])/self.L2Norms[0])
		axs[6].set_ylabel('rel. L2-Norm')
		axs[6].set_yscale('log')
		
		plt.savefig(f'{self.output_directory}/conservation.png')
		
		if show:
			plt.show()
		
		plt.close()
	
	def plot_E(self):
		
		fig, axs = plt.subplots(3)
		
		axs[0].plot(self.rho)
		axs[0].set_ylabel('Rho')
		
		axs[1].plot(self.phi)
		axs[1].set_ylabel('Phi')
		
		axs[2].plot(self.E)
		axs[2].set_ylabel('E')
		
		plt.show()
	
	def plot_feminusfi(self):
		
		print(self.fe.shape)
		fig, axs = plt.subplots(1, 3)
		
		axs[0].imshow(self.fe.T)
		axs[0].set_title('fe')
		
		axs[1].imshow(self.fi.T)
		axs[1].set_title('fi')
		
		axs[2].imshow((self.fe - self.fi).T)
		axs[2].set_title('fe-fi')
		
		plt.show()
	
	def plot_surf(self, Z, cbar = True):
		from matplotlib import cm
		X, Y = np.meshgrid(self.xs, self.vs)
		
		fig, ax = plt.subplots(subplot_kw = {'projection' : '3d'})
		surf = ax.plot_surface(X, Y, Z, cmap = cm.viridis, antialiased = False)
		if cbar:
			fig.colorbar(surf)
			
		plt.show()
	
	#output
	
	def output_data(self, txt=False, npy=True):
		#make output 
		
		fnames = ['fe', 'fi', 'E', 'Esq', 'Phi', 'Rho', 'Mass', 'Momentum', 'Ekin', 'Epot', 'Etotal', 'nE', 'ts', 'Entropy', 'L1Norm', 'L2Norm', 'xs', 'vs']\
		
		fnames_dict = {
						'fe' : self.fe, 'fi' : self.fi, 'E' : self.E, 'Phi' : self.phi, 'Rho' : self.rho,
						'Esq' : self.Esqs, 'ts' : self.ts, 
						'Mass' : self.Masses, 'nE' : self.nEs, 'Momentum' : self.Momentums, 
						'Ekin' : self.Ekins, 'Epot' : self.Epots, 'Etotal' : self.Etotals, 
						'Entropy' : self.Entropys, 'L1Norm' : self.L1Norms, 'L2Norm' : self.L2Norms, 
						'xs' : self.xs[1::2], 'vs' : self.vs[1::2]
						}

		
		dirpath = f'{self.output_directory}/data' 
		os.system(f'mkdir {dirpath}')
		
		if npy:
			for fname in fnames:
				np.save(dirpath + '/' + fname + '.npy', fnames_dict[fname])
			
		if txt:
			for fname in fnames:
				np.savetxt(dirpath + '/' + fname + '.dat', fnames_dict[fname])
		
		#write snapshots to seperate directory
		if self.snapshots:
			
			dirpath = f'{self.output_directory}/snapshots' 
			os.system(f'mkdir {dirpath}')
			
			for tup in self.snapshotsarr:
				st, arr = tup
				fname = f'snapshot_fe_t={st}'
				
				if npy:
					np.save(dirpath + '/' + fname + '.npy', arr)
					
				if txt:
					np.savetxt(dirpath + '/' + fname + '.dat', arr)
			
			for tup in self.snapshotspointsarr:
				st, arr = tup
				fname = f'snapshot_points_fe_t={st}'
				
				if npy:
					np.save(dirpath + '/' + fname + '.npy', arr)
					
				if txt:
					np.savetxt(dirpath + '/' + fname + '.dat', arr)
					
					
		#write info-file to same dir
		def write_code_to_txt():
			with open(__file__, 'r') as script_file:
				script_content = script_file.read()
	
	
			with open(f"{self.output_directory}/src.txt", "w") as output_file:
				output_file.write(script_content)
		
		write_code_to_txt()

def main():
	
	'''
	launches a single AF-VP run with the given set of parameters  
	'''
	
	pde = AF_Vlasov(scheme='slice', splitting='Strang', init_cond='LD', nx=64, nv=64, capture_timeseries = True,snapshots=True)
	'''
	pde : Simulation object
	
	---Input parameters---
	scheme (str) : select AF method 'slice' (1st order fluxintegral, sec. 3.3), 'fluxintegral' (2nd order fluxintegral, sec. 3.4) or 'sliceDD' (discrepancy distribution, sec. 3.5)'
	splitting (str) : select operator splitting 'Lie' (1st order, alg.2), 'Strang' (2nd order, alg.3) or 'Yoshida' (4th order, alg.4)
	init_cond (str) : select intial condition 'LD' (linear Landau damping), 'TS' (two-stream instability) or 'SLD' (strong Landau damping)
	nx (int) : resolution grid cells along x-direction
	nv (int) : resolution grid cells along v-direction
	capture_timeseries (bool) : if true, conserved quantities (moments like mass, momentum, energy etc.) and E^2 are computed during runtime
	snapshots (bool) : if true, snapshots of the distribution function are captured in intervals (interval determined by pde.st)
	
	default parameters from the pde.__init__ can be manually overwritten before calling pde.solve
	
	e.g.
	'''
	pde.tmax = 30.5 
	
	pde.solve() 
	

def timeseries_complete(case):
	
	'''
	launches a series of  longtime simulations for different schemes and resolutions given a initial condition 
	
	Usage:
	study of the conservational properties (sec. 5.4) and the long time/damping behaviour of the electric field (sec. 5.2)
	snapshots of the distribution function are taken in given time intervals (sec. 5.3)
	cross-sections through the snapshots were discussed in sec. 5.3 (Fig. 12)
	
	'''
	
	import gc
	schemes = ['slice', 'sliceDD', 'fluxintegral']
	
	if case == 'LD':
		nxsnvs = [(16,16), (32,32), (64,64), (32,64), (16,32), (128,128)]
		
		for scheme in schemes:
			for nxnv in nxsnvs:
				nx, nv = nxnv
				
				pde = AF_Vlasov(scheme=scheme, splitting='Strang', init_cond=case, nx=nx, nv=nv, capture_timeseries = True, snapshots=True)
				
				if nv < 64:
					pde.tmax = 50.5
				elif (nv == 64):
					pde.tmax = 80.5
				elif (nv == 128):
					pde.tmax = 100.5
				
				pde.tmax = 20.
				pde.solve()
				
				del pde 
				gc.collect()
				time.sleep(1)
	
	elif case == 'TS':
		nxsnvs = [(32,32), (64,64),(128,128)]
		
		for scheme in schemes:
			for nxnv in nxsnvs:
				nx, nv = nxnv
				
				pde = AF_Vlasov(scheme=scheme, splitting='Strang', init_cond=case, nx=nx, nv=nv, capture_timeseries = True, snapshots=True)
				
				pde.tmax = 100.5 
				
				pde.solve()
				
				del pde 
				gc.collect()
				time.sleep(1)
	
	elif case == 'SLD':
		nxsnvs = [(32,32), (64,64), (128,128)]
		
		for scheme in schemes:
			for nxnv in nxsnvs:
				nx, nv = nxnv
				
				pde = AF_Vlasov(scheme=scheme, splitting='Strang', init_cond=case, nx=nx, nv=nv, capture_timeseries = True, snapshots=True)
				
				pde.tmax = 100.5 
				
				pde.solve()
				
				del pde 
				gc.collect()
				time.sleep(1)
	
def convergence_complete(case):
	
	'''
	launches a series of simulations with varying resolutions to study the convergence of different AF methods and different splitting-schemes
	
	timeseries are not captured in this case
	distribution function at final time is manually saved after the run
	
	Usage:
	convergence study (sec. 5.1)
	
	'''
	
	import gc
	
	nxs = [512, 256, 128, 64, 32, 16]
	timesplittings = ['Yoshida', 'Strang']
	schemes = ['slice', 'sliceDD', 'fluxintegral']
	
	path = f'convergence_data_{case}'
	os.system(f'mkdir {path}')
	for scheme in schemes:
		for timesplitting in timesplittings:
			for n in nxs:
		
				pde = AF_Vlasov(scheme=scheme, splitting=timesplitting, init_cond=case, nx=n, nv=n, capture_timeseries = False)
				
				pde.solve()
		
				np.save(f'{path}/fe_{pde.stepX_method}_{pde.timesplitting}_{n}.npy', pde.fe[1::2, 1::2]) 
				print("saved", pde.t)
				
				
				del pde
				gc.collect() 
				time.sleep(3)
	

if __name__ == '__main__':
	main()
	#timeseries_complete('LD')
	#convergence_complete('TS')
