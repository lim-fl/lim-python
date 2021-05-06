from androguard.core.bytecodes.apk import APK
from pathlib import Path
import configparser

config = configparser.ConfigParser()
config.read("lim_code/generate_dataset.conf")


def lim_features(apk_filepath):
    name = None
    features = manifest_features(apk_filepath)
    if features:
        name = Path(apk_filepath).stem
    if features and is_preinstalled(apk_filepath):
        features.append('lim_preinstalled')
    if features and is_malware(apk_filepath):
        features.append('malware')
    return name, features


def is_preinstalled(apk_filepath):
    return is_option(apk_filepath, 'preinstalled')


def is_malware(apk_filepath):
    return is_option(apk_filepath, 'malware')


def is_option(apk_filepath, option):
    dirs = config.get('dataset', option).split(',')
    apk_path = Path(apk_filepath)
    return any([d in apk_path.parts for d in dirs])

def manifest_features(apk_filepath):
    try:
        apk = APK(apk_filepath)
        info = {'declared permissions': sorted(apk.get_permissions()),
            'activities': apk.get_activities(),
                'services': apk.get_services(),
                'intent filters': apk.get_intent_filters('receiver', ''),
                'content providers': apk.get_providers(),
                'broadcast receivers': apk.get_receivers(),
                'hardware components': apk.get_features()}

        return [item for key in info.keys() for item in info[key]]
    except:
        # We just do not process the APK
        pass
