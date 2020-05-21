import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')


def water_density_calculator(T, P, density=True):
    """ Estimate the density or molar volume of pure water at a specific
    T and P based on the compensated Redlich-Kwong model of Holland and
    Powell (1991)

    Parameters
    ----------
    T : positive scalar
        the temperature in C degrees

    P : positive scalar
        the pressure in GPa

    density : bool
       if True the function returns the water density, if False the molar volume

    Reference
    ---------
    Holland and Powell (1991) https://doi.org/10.1007/BF00306484

    Calls functions
    ---------------
    get_MKR_volume

    Returns
    -------
    water density in g/cm3 or molar volume in cm3/mol
    """

    # check model assumptions
    if T < 100 or T > 1600:
        raise ValueError('The temperature provided is out of the safe range for the model (100 - 1600)')
    elif P < 0.001 or P > 5:
        raise ValueError('The pressure provided is out of the safe range for the model (0.001 - 5 GPa)')

    # convert from GPa to kbar (P), and from C degrees to K (T)
    P = P * 10
    T = T + 273.15

    # set experimentally derived constans from Table 1 in Holland and Powell (1991)
    P0 = 2.0
    c = -3.025650e-2 - 5.343144e-6 * T
    d = -3.2297554e-3 + 2.2215221e-6 * T

    # estimate molar volume in cm3 per mol using the MKR equation
    if P > P0:
        V_mrk = get_MKR_volume(T, P)
        V = V_mrk + c * np.sqrt((P - P0) + d * (P - P0))
    else:
        V = get_MKR_volume(T, P)

    if density is True:
        return round(18.0152 / V, 4)
    else:
        return round(V, 4)


def get_MKR_volume(T, P, epsilon=0.0001, limit_guesses=100):
    """ The Modified Redlich-Kwong (MKR) equation is an empirically derived
    equation that estimate accurately the molar volume of water for pressures
    up to 5 GPa (50 kbar) in the T range 100-1400 deg C. The model proposed
    here is based on works by Redlich and Kwong (1949), Halbach and
    Chatterjee (1982) and Holland and Powell (1991). This function uses
    a bisection search algorithm to find the solution to the MKR equation.

    Parameters
    ----------
    T : positive scalar
        the temperature in K

    P : positive scalar
        the pressure in kbar

    epsilon : positive scalar (float), optional
        the accuracy of the estimate, default=0.0001

    limit_guesses : positive integer, optional
        the maximum number of attempts for estimation

    References
    ----------
    Halbach and Chatterjee (1982) https://doi.org/10.1007/BF00371526
    Holland and Powell (1991) https://doi.org/10.1007/BF00306484
    Redlich and Kwong (1949)

    Call function
    -------------
    CORK_equation

    Returns
    -------
    molar volume in cm3/mol
    """

    R = 8.3144598e-3  # universal gas constant [kJ / mol K]

    # estimate a and b constants. See Table 1 in Holland and Powell (1991)
    if T <= 673:
        #a = 1113.4 + 5.8487 * (673 - T) - 2.1370e-2 * (673 - T)**2 + 6.8133e-5 * (673 - T)**3
        a = 1113.4 + -0.88517 * (673 - T) + 4.5300e-3 * (673 - T)**2 - 1.3183e-5 * (673 - T)**3
    else:
        a = 1113.4 - 0.22291 * (T - 673) - 3.8022e-4 * (T - 673)**2 + 1.7791e-7 * (T - 673)**3

    b = 1.465  # kJ / kbar mol

    # find the molar volume using a bisection searh algorithm
    min_vol = -5
    max_vol = (R * T) / P + 10 * b
    num_guesses = 0
    guess_vol = (max_vol + min_vol) / 2
    result = 10

    while abs(result) >= epsilon:

        if num_guesses >= limit_guesses:
            print('The algorithm reached the maximum number of guesses without finding a solution!')
            print('Check the inputs (units) or try incresing the number of guesses (rarely useful)')
            print('last guess:', guess_vol)
            return None

        result = MKR_equation(T, P, guess_vol, a, b, R)

        if result > 0:
            max_vol = guess_vol
        else:
            min_vol = guess_vol

        guess_vol = (max_vol + min_vol) / 2
        num_guesses += 1

#    print('num of guesses:', num_guesses)

    # 1J/bar = 1kJ/kbar = 10cm3 = 10e-5m3
    return 10 * guess_vol


def MKR_equation(T, P, V, a, b, R):
    """ The modified Redlich-Kwong equation (MKR) of Halbach and Chatterjee
    (1982) rearranged by Holland and Powell (1991).

    Parameters
    ----------
    T : positive scalar
        the temperature in K

    P : positive scalar
        the pressure in kbars

    V : positive scalar
        the molar volume of water in cm3/mol

    a, b : scalars
        experimentally-derived constants (Holland and Powell, 1991)
        units:
            a: kJ**2 kbar**-1 K**1/2 mol**-2
            b: kJ kbar**-1 mol**-1

    R : scalar
        universal gas constant [kJ / mol K]

    Assumptions
    -----------
    first_term - second_term - third_term - fourth_term = 0
    """

    first_term = P * V**3
    second_term = R * T * V**2
    third_term = (b * R * T + b**2 * P - a / T**0.5) * V
    fourth_term = (a * b) / T**0.5

    return first_term - second_term - third_term - fourth_term


def test_one():
    """Reproduce the figures 2 and 3 of Holland and Powell to test the script"""

    func = np.vectorize(water_density_calculator)

    T_array = np.linspace(100, 1000)
    V_P1 = func(T_array, 0.1, density=False)
    V_P2 = func(T_array, 0.2, density=False)
    V_P5 = func(T_array, 0.5, density=False)
    V_P10 = func(T_array, 1.0, density=False)
    V_P50 = func(T_array, 5.0, density=False)

    P_array = np.linspace(0.4, 1.0)
    V_T800 = func(800, P_array, density=False)
    V_T600 = func(600, P_array, density=False)
    V_T400 = func(400, P_array, density=False)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax1.plot(P_array, V_T800, label='800 °C')
    ax1.plot(P_array, V_T600, label='600 °C')
    ax1.plot(P_array, V_T400, label='400 °C')
    ax1.set(xlabel='P (GPa)', ylabel='molar Vol (cm3/mol)', title='Figure 3')
    ax1.set_xlim(0.3, 1.1)
    ax1.set_ylim(16, 30)
    ax1.legend(fontsize=9)

    ax2.plot(T_array, V_P1, label='1 kbar')
    ax2.plot(T_array, V_P2, label='2 kbar')
    ax2.plot(T_array, V_P5, label='3 kbar')
    ax2.plot(T_array, V_P10, label='10 kbar')
    ax2.plot(T_array, V_P50, label='50 kbar')
    ax2.set(xlabel='T (C)', ylabel='molar Vol (cm3/mol)', title='Figure 2a')
    ax2.set_xlim(0, 1100)
    ax2.set_ylim(10, 80)
    ax2.legend(fontsize=9)

    return fig.tight_layout()


def test_two():
    """Make two plots: a T vs density, and a P vs density"""

    func = np.vectorize(water_density_calculator)

    T_array = np.linspace(100, 1000)
    V_P1 = func(T_array, 0.1)
    V_P2 = func(T_array, 0.2)
    V_P5 = func(T_array, 0.5)
    V_P10 = func(T_array, 1.0)
    V_P50 = func(T_array, 5.0)

    P_array = np.linspace(0.01, 1.0)
    V_T800 = func(800, P_array)
    V_T600 = func(600, P_array)
    V_T400 = func(400, P_array)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax1.plot(P_array, V_T800, label='800 °C')
    ax1.plot(P_array, V_T600, label='600 °C')
    ax1.plot(P_array, V_T400, label='400 °C')
    ax1.set(xlabel='P (GPa)', ylabel='density (g/cm3)')
    ax1.legend(fontsize=9)

    ax2.plot(T_array, V_P1, label='0.1 GPa')
    ax2.plot(T_array, V_P2, label='0.2 GPa')
    ax2.plot(T_array, V_P5, label='0.3 GPa')
    ax2.plot(T_array, V_P10, label='1 GPa')
    ax2.plot(T_array, V_P50, label='5 GPa')
    ax2.set(xlabel='T (C)', ylabel='density (g/cm3)')
    ax2.legend(fontsize=9)

    return fig.tight_layout()
