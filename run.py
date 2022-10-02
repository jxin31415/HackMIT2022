from predict_water_quality import init
from getJson import getData

state = 'nc'

getData(state)
init(0.0, 0.0, 2022, 5, 1, 365)
