from era5_api import get_era5

paramsList = [
    # '10m_u_component_of_wind',
    # '10m_v_component_of_wind',
    #  '2m_dewpoint_temperature',
    # '2m_temperature',
    #  'clear_sky_direct_solar_radiation_at_surface',
    # 'cloud_base_height',
    #  'convective_snowfall',
    #  'downward_uv_radiation_at_the_surface',
    #  'forecast_albedo',
    #  'forecast_logarithm_of_surface_roughness_for_heat',
    # 'high_cloud_cover',
    #  'instantaneous_surface_sensible_heat_flux',
    #  'large_scale_snowfall',
    # 'low_cloud_cover',
    # 'mean_snowfall_rate',
    # 'medium_cloud_cover',
    #  'near_ir_albedo_for_diffuse_radiation',
    #  'near_ir_albedo_for_direct_radiation',
    # 'snow_albedo',
    # 'snow_density',
    # 'snow_depth',
    # 'snowfall',
    #  'surface_latent_heat_flux',
    # 'surface_net_solar_radiation_clear_sky',
    'surface_net_solar_radiation',
    #  'surface_net_thermal_radiation',
    #  'surface_net_thermal_radiation_clear_sky',
    #  'surface_sensible_heat_flux',
    #  'surface_solar_radiation_downward_clear_sky',
    #  'surface_solar_radiation_downwards',
    #  'surface_thermal_radiation_downward_clear_sky',
    #  'surface_thermal_radiation_downwards',
    #  'toa_incident_solar_radiation',
    #  'top_net_solar_radiation',
    #  'top_net_solar_radiation_clear_sky',
    #  'top_net_thermal_radiation',
    #  'top_net_thermal_radiation_clear_sky',
    # # 'total_cloud_cover',
    #  'total_column_ozone',
    #  'total_precipitation',
    #  'total_sky_direct_solar_radiation_at_surface',
    #  'uv_visible_albedo_for_diffuse_radiation',
    #  'uv_visible_albedo_for_direct_radiation',
    # # 'surface_pressure',
]
data = {}
years = []

for i in range(2022, 2023, 1):
    years.append(i)

for params in paramsList:
    for n in years:
        data[str(n) + '-' + str(n + 1)] = \
            get_era5(var=params,
                     Year=[n, n + 1],
                     Month=[1, 12],
                     Day=[1, 31],
                     )
