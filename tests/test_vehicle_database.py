from co2mpas_driver import dsp as driver


def test_vehicle_database(db_path):
    """
        Check which vehicles work in the database.

    """

    # 44387 - electric problem
    veh_ids = [17146, 40516, 35452, 40225, 7897, 7972, 41388, 35361, 5768, 5766,
               3408, 15552, 9645, 9639, 8620, 8592, 5779, 5798, 8280, 8267,
               4396, 4416, 34271, 34265, 6378, 39723, 34092, 2508, 2592, 5635,
               5630, 34499, 34474, 7661, 7683, 8709, 9769, 20409, 10133, 26765,
               1872, 10328, 10349, 35476, 41989, 26799, 26851, 27189, 27096,
               23801, 3079, 36525, 47766, 6386, 6390, 18771, 18767, 2090, 1978,
               33958, 33986, 5725, 5718, 36591, 4350, 39396, 40595, 5909, 5897,
               5928, 5915, 40130, 42363, 34760, 34766, 1840, 1835, 36101, 42886,
               1431, 24313, 46547, 44799, 41045, 39820, 3231, 3198, 34183,
               34186, 20612, 20605, 1324, 9882, 9885, 4957, 44856, 18195, 5595,
               5603, 18831, 18833, 22376, 9575, 5391, 5380, 9936, 7995, 6331,
               18173, 43058, 34286, 34279, 20699, 20706, 34058, 34057, 24268,
               24288, 19028, 19058, 7979, 7976, 22563, 22591, 34202, 34196,
               40170, 44599, 5358, 5338, 34024, 34015, 7836, 7839, 9738, 9754,
               9872, 9856, 6446, 8866, 9001, 8996, 9551, 6222, 35135, 39393,
               27748, 15109, 8183, 8188, 26629, 8358, 17145]

    # veh_ids = [39393, 8188, 40516, 35452, 40225, 7897, 7972, 41388, 5766, 9645,
    #            9639, 5798, 8280, 34271, 34265, 6378, 39723, 34092, 2592, 5635,
    #            5630, 7661, 7683]

    problem_ids = []
    ok_ids = []
    complete = 0
    cnt = 0
    while complete == 0:
        try:
            test_ids = veh_ids[cnt:cnt]
            vehicles = [driver(dict(vehicle_id=i, db_path=db_path,
                         inputs=dict(inputs={'gear_shifting_style': 0.7,
                                             'driver_style': 1})))
             ['outputs']['driver_simulation_model'] for i in test_ids]
            ok_ids.append(veh_ids[cnt])
        except:
            problem_ids.append(veh_ids[cnt])
        cnt += 1
        if cnt == len(veh_ids):
            complete = 1

    print("Vehicles which are fine: ", ok_ids)
    print("Vehicles which doesn't work: ", problem_ids)

    return 0


if __name__ == '__main__':
    db_path = '../co2mpas_driver/db/EuroSegmentCar_cleaned.csv'
    test_vehicle_database(db_path)