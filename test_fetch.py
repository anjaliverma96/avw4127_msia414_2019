import requests
def test():
	PARAMS = {'plot' : "Only a few years after the American geologist Adrian Helmsley's warnings of an impending global Armageddon by the year 2012, the Earth is devastated from end to end by cataclysmic natural disasters. As the President of the U.S. along with other leaders of the G8 Nations complete their secret project in Tibet to build colossal arks to sustain humanity, at the same time, the struggling Los Angeles author, Jackson Curtis, goes through hell and back to reunite with his ex-wife and their two kids. Inevitably, the unfathomable catastrophes are rapidly escalating, while Jackson strives to give his family a future in Tibet, however, can he make it in time?"}
	r = requests.post(url = 'http://127.0.0.1:5000/get_predictions', params = PARAMS)
	print('HERE',r.json())

test()
