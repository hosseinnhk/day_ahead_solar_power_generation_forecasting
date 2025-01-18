def get_era5(dataset_name='reanalysis-era5-single-levels',
             var=None,
             Year=None,
             Month=[1, 12],
             Day=[1, 31],
             Time=[
                 '00:00', '01:00', '02:00',
                 '03:00', '04:00', '05:00',
                 '06:00', '07:00', '08:00',
                 '09:00', '10:00', '11:00',
                 '12:00', '13:00', '14:00',
                 '15:00', '16:00', '17:00',
                 '18:00', '19:00', '20:00',
                 '21:00', '22:00', '23:00',
             ],
             # area=[59.4, 24.65, 59.38, 24.67], # Tallinn
             # area=[40.59, -3.94, 40.57, -3.92],  # Madrid
             # area=[41.89, 12.48, 41.91, 12.50],  #Rome
             # area=[50.84, 4.36, 50.86, 4.38], #Brussel
             # area=[59.91, 10.74, 59.93, 10.76], #Oslo
             # area=[48.15, 11.58, 48.17, 11.60],  # Munich
             area=[59.09, 24.29, 59.11, 24.31],  # pv farm tallinn

             ):
    import cdsapi
    import xarray as xr
    import pandas as pd
    from urllib.request import urlopen

    c = cdsapi.Client()

    yearList = list(range(Year[0], Year[1] + 1))
    yearStrList = []
    for item in yearList:
        yearStrList.append(str(item))

    monthList = list(range(Month[0], Month[1] + 1))
    monthStrList = []
    for item in monthList:
        monthStrList.append(str(item))

    dayList = list(range(Day[0], Day[1] + 1))
    dayStrList = []
    for item in dayList:
        dayStrList.append(str(item))

    timeList = Time

    # parameters
    params = dict(
        format="netcdf",
        product_type="reanalysis",
        variable=var,
        area=area,
        year=yearStrList,
        month=monthStrList,
        day=dayStrList,
        time=timeList,
    )

    # file object
    fl = c.retrieve(dataset_name, params)

    # download the file
    download_file = './' + str(var) + '_' + yearStrList[0] + '_' + yearStrList[-1] + '.nc'
    fl.download(f"{download_file}")

    # load into memory and return xarray dataset
    with urlopen(fl.location) as f:
        return xr.open_dataset(f.read())
