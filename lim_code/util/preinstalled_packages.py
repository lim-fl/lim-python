"""Generates a dictionary of preinstalled apps.

The input file has to have the list of system packages as per adb.

The command to generate it is 'adb shell cmd package list packages -s -f'.
"""

import re
import subprocess

input_file = 'preinstalled_packages.txt'
google_patterns = ['^com.android.*', '^com.google']
google_regexp = re.compile('|'.join((google_patterns)))
apk_directory = './preinstalled_apks'


def app_line(line):
    """Return a list [packagename, apk].

    It expects lines with the format package:apk=packagename.
    """
    if line:
        aux_apk, packagename = line.split('=')
        apk = aux_apk.split(':')[1]
        return [packagename, apk]


def apps(input_file):
    """Get a dictionary with the package names and the its apk path."""
    with open(input_file) as f:
        lines = f.read().split('\n')
        _list_apps = [app_line(line) for line in lines if line]
        apps = {app[0]: app[1] for app in _list_apps}
        return apps


def google_apps(input_file):
    _apps = apps(input_file)
    return {key: _apps[key] for key in _apps.keys()
            if google_regexp.match(key)}


def extract_apks_with_adb():
    apps = google_apps(input_file)
    for packagename in apps.keys():
        apk = apps[packagename]
        subprocess.run(f'adb pull {apk} {apk_directory}'.split(' '))


def main():
    print(google_apps(input_file))


if __name__ == '__main__':
    main()
