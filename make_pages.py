import json
from os import listdir


# add more content
def write_markdown(name, description):
    with open(f'{name}.markdown', 'w') as f:
        # metadata
        f.write(f'--- \nlayout: page \ntitle: {name} \npermalink: /{name} \n--- \n \n')

        # title
        f.write(f'# {name} \n--- \n \n')

        # description
        f.write("## Description \n")
        f.write(description)
        f.write("\n\n---")

        # dataset link, models, notebooks etc.


datasets = listdir('datasets')

for json_file in datasets:
    with open('datasets/' + json_file, 'r') as f:
        data = json.load(f)
        write_markdown(data['name'], data['description'])
