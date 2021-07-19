import json
from os import listdir


# add more content
def write_markdown(name, description):
    with open(f'{name}.md', 'w') as f:
        f.write(f'-- \n layout: page \n title: {name} \n permalink: /datasets/{name} \n -- \n \n')

        f.write("## Description \n \n ")
        f.write(description)


datasets = listdir('datasets')

for json_file in datasets:
    with open('datasets/' + json_file, 'r') as f:
        data = json.load(f)
        write_markdown(data['name'], data['description'])
