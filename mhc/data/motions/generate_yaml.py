import os

dataset = 'sfu'

file_dir = os.path.dirname(os.path.abspath(__file__))
source_motion_dir = os.path.join(file_dir, dataset)
yaml_file_name = os.path.join(source_motion_dir, "dataset_"+dataset+'.yaml')
os.remove(yaml_file_name)
source_motion_files = sorted(os.listdir(source_motion_dir))

weight = 1./len(source_motion_files)

text = 'motions:\n'
for file_name in source_motion_files:
    text += '  - file: \"' + file_name + '\"\n'
    text += '    weight: ' + str(weight) + '\n'

with open(yaml_file_name, 'w') as fh:
    fh.writelines(text)