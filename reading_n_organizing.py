import vehicle_characteristic_class as vcc

def load_db_to_dictionary(name):
    '''
    input: file (without .csv)
    return: dict of dictionaries, every dict item is a car, every dict key is the car's characteristic's name, and the value returned from the key is the value of the characteristic
    '''
    ##First replace any , with - in the csv
    file = open(name+'.csv', 'r', encoding="ISO-8859-1")
    A = file.readline()
    Characteristics = []
    Out = {}

    ##create the list of the names of the characteristic
    while A != '':
        k = A.find(',')
        if k == -1:
            Characteristics.append(A)
            A = ''
        else:
            Characteristics.append(A[:k])
            A = A[k + 1:]

    ##for every line of the csv creates a dictionary
    ##then for every cell of the line, stores the value to the dictionary using the characteristic's name as key
    for line in file:
        C = {}
        for i in range(len(Characteristics) - 1):
            k = line.find(',')
            C[Characteristics[i]] = line[:k]
            line = line[k + 1:]
        C[Characteristics[-1].rstrip('\n')] = line.rstrip('\n')
        Out[int(C[Characteristics[0]])] = C

    file.close()
    return Out

def get_vehicle_from_db(db_dict,car_id,**kwargs):
    '''

    kwargs can be:
    lco = True          # Light co2mpas is to be used, so the relevant parameters must be imported
    electric = True     # The vehicle is an EV

    :param db_dict: dict of dicts ,A dictionary produced by the "load_db_to_dictionary@" function
    :param car_id: int, The car id in the db
    :return: veh_specs object
    '''

    my_car = db_dict[car_id]

    my_car_specs = vcc.veh_specs(my_car,**kwargs)

    return my_car_specs