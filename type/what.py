import json, sys
from pprint import pprint

args = sys.argv
with open(str(args[1])) as file:
    data = json.load(file)

#pprint(data)

print(data['characteristics']['vis_int'])
