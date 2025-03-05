"""
Copyright (c) 2025 CyberCortex Robotics SRL. All rights reserved
CyberCortex.AI.dojo: neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (contact@cybercortex.ai)
"""

"""
Script for generating the map between two object classes confs.
"""

import argparse
import os
import io, libconf


def create_map(args):
    """
       Paths:
        {
            Old = "<old_path>.conf"
            New = "<new_path>.conf"
        },
       Mapping:
        {
            <old_name> :
            {
                ID     = <old_id>
                New_ID = <new_id>  # <new_name>
            }
            <old_name> :
            {
                ID     = <old_id>
                New_ID = <new_id>  # <new_name>
            }
            ...
        }
    """

    # Read old object classes
    assert os.path.exists(args.old_cls), "Old object classes does not exist."
    with io.open(args.old_cls) as f:
        old_cls = libconf.load(f)['ObjectClasses']

    # Read new object classes
    assert os.path.exists(args.new_cls), "New object classes does not exist."
    with io.open(args.new_cls) as f:
        new_cls = libconf.load(f)['ObjectClasses']

    map_file = 'Paths:\n{\n'
    map_file += '    Old = ' + '\"{}\"\n'.format(args.old_cls.replace('\\', '/'))
    map_file += '    New = ' + '\"{}\"\n'.format(args.new_cls.replace('\\', '/'))
    map_file += '},\nMapping:\n{\n'

    # Clear console
    def cls():
        # os.system('cls')
        print('\n' * 2)

    print('\nNew classes will be displayed along with one old class.')
    print('Type the id of the new class you want to map with the current old one.')
    input('Press Enter to continue..')

    new_id = 0
    new_key = ''
    for old_key in old_cls.keys():
        found = False

        while not found:
            cls()  # Clear screen

            # Print new classes
            print(' '.join(['{}:{},'.format(new_cls[name].ID, name) for name in new_cls.keys()]))
            print('Old class: ' + old_key)
            new_id = input('New id: ')

            if not new_id.isnumeric():
                continue

            # Find new name from the new id
            for new_key in new_cls.keys():
                if new_id == str(new_cls[new_key].ID):
                    found = True
                    break

        old_id = str(old_cls[old_key].ID)
        map_file += '    ' + old_key
        map_file += ':\n    {\n'
        map_file += '        ID     = ' + old_id + '\n'
        map_file += '        New_ID = ' + new_id + '  # ' + new_key
        map_file += '\n    }\n'
    map_file += '}'

    with open(args.map_path, 'w') as f:
        f.write(map_file)

    print('Mapping conf file written successfully.')


def main():
    parser = argparse.ArgumentParser(description='Script for creating map conf for class interchange.')

    parser.add_argument('--old_cls', help='Path to the old object classes file.',
                        default=r'C:\data\ECP_converted\datastream_2\object_classes.conf')
    parser.add_argument('--new_cls', help='Path to the new object classes file.',
                        default=r'C:\dev\src\CyberCortex.AI\dojo\etc\env\object_classes_counter_2.conf')
    parser.add_argument('--map_path', help='Path to the cls mapping conf.',
                        default=r'C:\data\cls_map.conf')

    args = parser.parse_args()

    create_map(args=args)


if __name__ == '__main__':
    main()
