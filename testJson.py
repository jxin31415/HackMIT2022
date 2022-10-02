import json
import urllib.request
import pandas as pd

targetVariables = {
    'Temperature, water, degrees Celsius' : 0,
    'Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius': 1,
    'Dissolved oxygen, water, unfiltered, milligrams per liter': 2,
    'pH, water, unfiltered, field, standard units': 3,
    'Turbidity, water, unfiltered, monochrome near infra-red LED light, 780-900 nm, detection angle 90 +-2.5 degrees, formazin nephelometric units (FNU)': 4
}

def getData():
    with urllib.request.urlopen("https://waterservices.usgs.gov/nwis/dv/?format=json&indent=on&stateCd=co&period=P10W&siteStatus=all") as url:
        data = json.load(url)
        data = data['value']['timeSeries']

        ret = {}
        for sample in data:
            source = sample['sourceInfo']['geoLocation']['geogLocation']
            variable = sample['variable']['variableDescription'] 
            values = sample['values'][0]['value']
            
            for val in values:

                input = (
                    val['dateTime'],
                    source['latitude'],
                    source['longitude']
                )

                cur = ret.get(input)
                if not cur:
                    ret[input] = {}

                ret[input][variable] = val['value']

        data = {}
        i = 0
        for key in ret.keys():
            i = i + 1
            row = [key[0], key[1], key[2], "", "", "", "", ""]
            for variable in ret[key].keys():
                if not targetVariables.get(variable) is None:
                    val = targetVariables.get(variable) + 3
                    row[val] = ret[key][variable]

            if row[3] or row[4] or row[5] or row[6] or row[7]:
                data[i] = row
        
        dataFrame = pd.DataFrame.from_dict(data, orient='index', columns=['dateTime', 'latitude', 'longitude', 'temperature', 'conductance', 'dissolved oxygen', 'pH', 'turbidity'])

        with open("exampleJSON.json", "w") as outfile:
            outfile.write(dataFrame.to_string())

        return dataFrame



getData()