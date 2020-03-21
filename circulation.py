import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Circulation:
    """
    Model of systemic circulation from Ferreira et al. (2005), A Nonlinear State-Space Model
    of a Combined Cardiovascular System and a Rotary Pump, IEEE Conference on Decision and Control.
    """

    def __init__(self, HR, Emax, Emin, R1=1.0, R3=0.001):
        self.set_heart_rate(HR)

        self.Emin = Emin
        self.Emax = Emax
        self.non_slack_blood_volume = 250 # ml

        self.R1 = R1 # between .5 and 2
        self.R2 = .005
        self.R3 = R3
        self.R4 = .0398

        self.C2 = 4.4
        self.C3 = 1.33

        self.L = .0005

    def set_heart_rate(self, HR):
        """
        Sets several related variables together to ensure that they are consistent.

        :param HR: heart rate (beats per minute)
        """
        self.HR = HR
        self.tc = 60/HR
        self.Tmax = .2+.15*self.tc # contraction time

    def get_derivative(self, t, x):
        """
        :param t: time
        :param x: state variables [ventricular pressure; atrial pressure; arterial pressure; aortic flow]
        :return: time derivatives of state variables
        """

        """
        WRITE CODE HERE
        Implement this by deciding whether the model is in a filling, ejecting, or isovolumic phase and using 
        the corresponding dynamic matrix. 
         
        As discussed in class, be careful about starting and ending the ejection phase. One approach is to check 
        whether the flow is >0, and another is to check whether x1>x3, but neither will work. The first won't start 
        propertly because flow isn't actually updated outside the ejection phase. The second won't end properly 
        because blood inertance will keep the blood moving briefly up the pressure gradient at the end of systole. 
        If the ejection phase ends in this time, the flow will remain non-zero until the next ejection phase. 
        """

        if x[1] > x[0]:
            a_mat = self.filling_phase_dynamic_matrix(t)
        elif (x[3] > 0 or x[0] > x[2]):
            a_mat = self.ejection_phase_dynamic_matrix(t)
        else:
            a_mat = self.isovolumic_phase_dynamic_matrix(t)

        return np.matmul(a_mat,x)


    def isovolumic_phase_dynamic_matrix(self, t):
        """
        :param t: time (s; needed because elastance is a function of time)
        :return: A matrix for isovolumic phase
        """
        el = self.elastance(t)
        del_dt = self.elastance_finite_difference(t)
        return [[del_dt/el, 0, 0, 0],
             [0, -1/(self.R1*self.C2), 1/(self.R1*self.C2), 0],
             [0, 1/(self.R1*self.C3), -1/(self.R1*self.C3), 0],
             [0, 0, 0, 0]]

    def ejection_phase_dynamic_matrix(self, t):
        """
        :param t: time (s)
        :return: A matrix for filling phase
        """
        el = self.elastance(t)
        del_dt = self.elastance_finite_difference(t)
        return [[del_dt/el, 0, 0, -el],
                [0, -1/(self.R1*self.C2), 1/(self.R1*self.C2), 0],
                [0, 1/(self.R1*self.C3), -1/(self.R1*self.C3), 1/self.C3],
                [1/self.L, 0, -1/self.L, -(self.R3+self.R4)/self.L]]

    def filling_phase_dynamic_matrix(self, t):
        """
        :param t: time (s)
        :return: A matrix for filling phase
        """

        """
        WRITE CODE HERE
        """
        el = self.elastance(t)
        del_dt = self.elastance_finite_difference(t)
        return [[del_dt/el - el/self.R2, el/self.R2, 0, 0],
                [1/(self.R2*self.C2), -(self.R1+self.R2)/(self.R1*self.R2*self.C2), 1/(self.R1*self.C2), 0],
                [0, 1/(self.R1*self.C3), -1/(self.R1*self.C3), 0],
                [0, 0, 0, 0]]


    def elastance(self, t):
        """
        :param t: time (needed because elastance is a function of time)
        :return: time-varying elastance
        """
        tn = self._get_normalized_time(t)
        En = 1.55 * np.power(tn/.7, 1.9) / (1 + np.power(tn/.7, 1.9)) / (1 + np.power(tn/1.17, 21.9))
        return (self.Emax-self.Emin)*En + self.Emin

    def elastance_finite_difference(self, t):
        """
        Calculates finite-difference approximation of elastance derivative. In class I showed another method
        that calculated the derivative analytically, but I've removed it to keep things simple.

        :param t: time (needed because elastance is a function of time)
        :return: finite-difference approximation of time derivative of time-varying elastance
        """
        dt = .0001
        forward_time = t + dt
        backward_time = max(0, t - dt) # small negative times are wrapped to end of cycle
        forward = self.elastance(forward_time)
        backward = self.elastance(backward_time)
        return (forward - backward) / (2*dt)

    def simulate(self, total_time):
        """
        :param total_time: seconds to simulate
        :return: time, state (times at which the state is estimated, state vector at each time)
        """

        """
        WRITE CODE HERE
        Put all the blood pressure in the atria as an initial condition.
        """

        ic = np.array([0,self.non_slack_blood_volume/self.C2,0,0])
        sol = solve_ivp(self.get_derivative, [0, total_time], ic, max_step=0.01, rtol=1e-5, atol=1e-8)

        return [sol.t, sol.y]


    def _get_normalized_time(self, t):
        """
        :param t: time
        :return: time normalized to self.Tmax (duration of ventricular contraction)
        """
        return (t % self.tc) / self.Tmax


    def get_left_vbv(self, t, vpt):
        """
        Question 3
        Calculate left ventricular blood volume given results from a simulation of the
        cardiovascular system model. Assume a slack ventricular volume of 20mL.
        Use elastance equation, E = P/(V - V0)

        :param t: time
        :param vpt: ventricular pressure at time t
        :return: ventricular blood volume at time t
        """
        el = self.elastance(t)
        slack_vol = 20  # mL

        return (vpt/el) + slack_vol


########################################################################################################################

def plot_q2():
    # Instantiate Circulation instance with given const
    circ = Circulation(75, 2.0, 0.06)

    # Simulate for 5 seconds
    [time, states] = circ.simulate(5.0)

    left_ventricular_P = states[0,:]
    atrial_P = states[1,:]
    arterial_P = states[2,:]
    aortic_flow_rate = states[3,:]

    # Aortic pressure right outside of aortic valve (btwn D2 and R4)
    aortic_P = arterial_P + aortic_flow_rate*circ.R4
    # aortic_P = left_ventricular_P - aortic_flow_rate*circ.R3

    plt.figure()
    plt.plot(time, left_ventricular_P)
    plt.plot(time, atrial_P)
    plt.plot(time, arterial_P)
    plt.plot(time, aortic_P)
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.title('Various Pressures vs. Time')
    plt.legend(('Ventricular Pressure, x1',
                'Atrial Pressure, x2',
                'Arterial Pressure, x3',
                'Aortic Pressure btwn D2 and R4'),
               loc='upper left')
    plt.show()


def plot_pv_loops():
    # Question 4
    # Plot ventricular pressure (y) vs ventricular volume (x) to produce
    # pressure-volume loop for a few cardiac cycles
    # Increase R1 to 2 Ohms
    # Use R1 = 0.5 and R3 = 0.2

    # Instantiate the three Circulation models
    circ_normal = Circulation(75, 2.0, 0.06)
    circ_high_sys_res = Circulation(75, 2.0, 0.06, R1=2.0)
    circ_aortic_stenosis = Circulation(75, 2.0, 0.06, R1=0.5, R3=0.2)

    # Simulate for 5 seconds
    [t_norm, states_norm] = circ_normal.simulate(5.0)
    [t_high_res, states_high_res] = circ_high_sys_res.simulate(5.0)
    [t_aortic_stenosis, states_aortic_stenosis] = circ_aortic_stenosis.simulate(5.0)

    # Get ventricular pressures for respective Circulation models
    vp_norm = states_norm[0,:]
    vp_high_res = states_high_res[0,:]
    vp_aortic_stenosis = states_aortic_stenosis[0,:]

    # Get ventricular volumes for respective Circulation models
    vv_norm = circ_normal.get_left_vbv(t_norm, vp_norm)
    vv_high_res = circ_high_sys_res.get_left_vbv(t_high_res, vp_high_res)
    vv_aortic_stenosis = circ_aortic_stenosis.get_left_vbv(t_aortic_stenosis, vp_aortic_stenosis)

    # Plot pressure-volume loops
    plt.figure()
    plt.plot(vv_norm[-165:],vp_norm[-165:])
    plt.plot(vv_high_res[-165:],vp_high_res[-165:])
    plt.plot(vv_aortic_stenosis[-165:],vp_aortic_stenosis[-165:])
    plt.xlabel('Ventricular Volume (mL)')
    plt.ylabel('Ventricular Pressure (mmHg)')
    plt.title('Pressure-Volume Loops for Various Cardiac Conditions')
    plt.legend(('Normal, R1 = 1 mmHg s/mL',
                'High systemic resistance, R1 = 2 mmHg s/mL',
                'Aortic stenosis, R1 = 0.5 mmHg s/mL, R3 = 0.2 mmHg s/mL'),
               loc='upper left')
    plt.tight_layout()
    plt.show()


# Question 2
plot_q2()

# Question 4
plot_pv_loops()