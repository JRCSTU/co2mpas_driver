import numpy as np

def find_car_gear(my_car, speed, rpm):
    '''

    Use speed and rpm to find the gear

    speed: m/s
    :return:
    '''

    gr_str = np.array([ 1/float(i)/my_car.final_drive for i in my_car.gr ])

    ratio = speed*60 / my_car.wheelbase / rpm

    res = np.abs(gr_str-ratio)

    return res.argmin()

def find_list_of_tans_from_coefs(coefs_per_gear, Start, Stop):

    '''

    Full load curve is fitted to a polynomial of degree

    :param speed_per_gear:
    :param acc_per_gear:
    :param degree:
    :return: coefs_per_gear: the coefficients of the polynomial for each gear
    '''

    degree = len(coefs_per_gear[0]) - 1
    vars = np.arange(degree, -1, -1)

    Tans = []

    for gear,coefs in enumerate(coefs_per_gear):

        x_new = np.arange(Start[gear], Stop[gear], 0.1)
        a_new = np.array([np.dot(coefs, np.power(i, vars)) for i in x_new])

        Tans.append(np.diff(a_new)*10)

    return Tans

def find_gs_cut_tans(tmp_min, tmp_max, tan, tmp_min_next, cutoff):

    max_tan = np.max(tan)
    min_tan = np.min(tan)
    acc_range = max_tan - min_tan

    # tan starts from positive and goes negative, so I use (1 - cutoff) for the percentage
    if cutoff > 0.99:
        cutoff = 1
    elif cutoff < 0.01:
        cutoff = 0.01
    tan_cutoff = (1 - cutoff) * acc_range + min_tan

    # Search_from = int(tmp_min_next * 10)
    Search_from = int((tmp_min_next - tmp_min)*10)+1

    i_cut = len(tan)-1
    while tan[i_cut]< tan_cutoff and i_cut >= Search_from:
        i_cut -=1

    gear_cut = tmp_min + i_cut/10 +0.1

    return gear_cut

def gear_points_for_AIMSUN_tan(Tans,gs_style,Start,Stop):

    n_gears = len(Tans)
    gs_cut = [gs_style for i in range(n_gears)]

    gs = []

    for i in range(n_gears - 1):
        tmp_min = Start[i]
        tmp_max = Stop[i]
        tan = Tans[i]
        tmp_min_next = Start[i+1]
        cutoff_s = find_gs_cut_tans(tmp_min, tmp_max, tan, tmp_min_next, gs_cut[i])

        gs.append(cutoff_s)

    return gs

def gear_for_speed_profiles(gs, curr_speed, current_gear, gear_cnt, automatic=0):
    ####THIS IS MODEL PARAMETERS FOR UPSHIFT AND DOWNSHIFT
    upshift_offs = 0.0  ####CHANGED FROM 0.05 TO 0
    downshift_off = 0.1

    gear_limits = [0]
    gear_limits.extend(gs)  ###maybe this should be done earlier to save memory
    gear_limits.append(200)

    if gear_limits[current_gear - 1] - gear_limits[current_gear - 1] * downshift_off <= curr_speed < gear_limits[
        current_gear] + gear_limits[current_gear] * upshift_offs:
        if gear_cnt == 0:
            return current_gear, gear_cnt
        else:
            gear_cnt -= 1
            if automatic == 1:
                # reduce acceleration due to GS
                return current_gear, gear_cnt
            else:
                return current_gear, gear_cnt
    else:
        iter = 1
        gear_search = 1
        while iter == 1:
            if gear_limits[gear_search - 1] <= curr_speed < gear_limits[gear_search]:
                gear_cnt = 5  # in simulation steps for 0.5 second
                current_gear = gear_search
                iter = 0
            else:
                gear_search += 1
        if automatic == 1:
            # reduce acceleration due to GS
            return current_gear, gear_cnt
        else:
            return current_gear, gear_cnt
