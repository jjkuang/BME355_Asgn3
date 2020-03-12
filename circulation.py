import numpy as np
from scipy.integrate import solve_ivp


class Circulation:
    """
    Model of systemic circulation from Ferreira et al. (2005), A Nonlinear State-Space Model
    of a Combined Cardiovascular System and a Rotary Pump, IEEE Conference on Decision and Control.
    """

    def __init__(self, HR, Emax, Emin):
        self.set_heart_rate(HR)

        self.Emin = Emin
        self.Emax = Emax
        self.non_slack_blood_volume = 250 # ml

        self.R1 = 1.0 # between .5 and 2
        self.R2 = .005
        self.R3 = .001
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

        xdot = []
        filling_phase = False
        ejection_phase = False
        isovolumic_phase = False

        if x[1] > x[0]:
            filling_phase = True
        elif x[4] > 0 or x[1] > x[3]:
            ejection_phase = True
        else:
            isovolumic_phase = True

        el = self.elastance(t)
        del_dt = self.elastance_finite_difference(t)
        if filling_phase:

            xdot[0] = del_dt*x[0]/el + (x[1]-x[0])/(self.R2*self.C1)
            xdot[1] = (x[2]-x[1])/(self.R1*self.C2) - (x[1]-x[0])/(self.R2*self.C2)
            xdot[2] = -(x[2]-x[1])/(self.R1*self.C3)
            xdot[3] = 0

        elif ejection_phase:

            xdot[0] = del_dt*x[0]/el - el*x[3]
            xdot[1] = (x[2]-x[1])/(self.R1*self.C2)
            xdot[2] = (x[3]/self.C3) - (x[2]-x[1])/(self.R1*self.C3)
            xdot[3] = ((x[0]-x[2])-x[3]*(self.R3+self.R4))/self.L

        elif isovolumic_phase:

            xdot[0] = del_dt*x[0]/el
            xdot[1] = (x[2]-x[1])/(self.R1*self.C2)
            xdot[2] = -xdot[1]
            xdot[3] = 0


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
                [1/(self.R1*self.C2), -(self.R1+self.R2)/(self.R1*self.R2*self.C2), 1/(self.R1*self.C2), 0],
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

        lm = np.arange(0, 1.8, .01)
        vm = np.arange(-1.2, 1.2, .01)
        lt = np.arange(0, 1.07, .01)
        plt.subplot(2,1,1)
        # plt.plot(lm, force_length_muscle(lm), 'r')
        # plt.plot(lm, force_length_parallel(lm), 'g')
        # plt.plot(lt, force_length_tendon(lt), 'b')
        # plt.legend(('CE', 'PE', 'SE'))
        plt.xlabel('Normalized length')
        plt.ylabel('Force scale factor')
        plt.subplot(2, 1, 2)
        # plt.plot(vm, force_velocity_muscle(vm), 'k')
        plt.xlabel('Normalized muscle velocity')
        plt.ylabel('Force scale factor')
        plt.tight_layout()
        plt.show()


    def _get_normalized_time(self, t):
        """
        :param t: time
        :return: time normalized to self.Tmax (duration of ventricular contraction)
        """
        return (t % self.tc) / self.Tmax



    def calc_left_vbv(self):
        """
        Question 4
        Calculate left ventricular blood volume given results from a simulation of the
        cardiovascular system model. Assume a slack ventricular volume of 20mL.

        :param t: time
        :return: time normalized to self.Tmax (duration of ventricular contraction)
        """
        return 0


########################################################################################################################
if __name__ == "__main__":

    # Question 1
    # circ = Circulation(75, 2.0, 0.06)


    # Question 2
    # circ.simulate(5)


    # Question 3