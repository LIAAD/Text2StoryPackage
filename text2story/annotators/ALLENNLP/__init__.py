import os
import shlex
from pathlib import Path

import subprocess
import platform
import uuid

import joblib
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

#from text2story.core.utils import is_library_installed, find_target_dir
from core.utils import is_library_installed, find_target_dir

current_path = Path(__file__).parent
# Path to the virtual environment's activation script
env_name = os.path.join(current_path, "allennlp_venv")
allen_script = os.path.join(current_path, "allen_wrapper.py")

def setup_enviroment():

    # Check if the VIRTUAL_ENV environment variable is set
    if 'VIRTUAL_ENV' in os.environ:
        # Extract the virtual environment name from the path
        env_path = os.environ['VIRTUAL_ENV']
        print(f"Current virtual environment: {os.path.basename(env_path)}")
    else:
        print("No virtual environment is currently activated.")

    # this model requires a older version of transformers library, so
    # this code test it and installed a newer version in a separate directory

    if not(os.path.exists(env_name)):
        # Create a virtual environment with the specified name
        subprocess.call(['python3', '-m', 'venv', env_name])

    # run the setup environment script
    # Determine the appropriate activation script based on the current platform
    if platform.system() == 'Windows':
        activation_script = os.path.join(current_path, "scripts", 'setup_env.bat')
        subprocess.call([activation_script, env_name], shell=True)
    else:
        target_dir = find_target_dir(env_name, "site-packages")

        if not(is_library_installed("allennlp", target_dir)):
            #command = f'source {env_name}/bin/activate && pip install -r {current_path}/requirements.txt'
            command = f'pip install --target={target_dir} -r {current_path}/requirements.txt'
            subprocess.call(command, shell=True,  executable="/bin/bash")


def load(lang):
    # this install the enviroment to run the allen nlp wrapper in
    # the new enviroment
    setup_enviroment()

def extract_participants(lang, text):

    venv_activation_script = os.path.join(env_name, "bin", "activate")

    # Generate a random UUID
    unique_id = str(uuid.uuid4())
    input_file = os.path.join(current_path, f"participant_{unique_id}.txt")
    with open(input_file, "w") as fd:
        fd.write(text)

    output_file = os.path.join(current_path, f"participant_output_{unique_id}.joblib")
    subprocess.call(f"source {venv_activation_script} && python {allen_script} --lang {lang} --action participant \
    --input {input_file} --output {output_file}", shell=True,  executable="/bin/bash")

    result = joblib.load(output_file)

    os.remove(output_file)
    os.remove(input_file)

    return result

def extract_events(lang, text):
    venv_activation_script = os.path.join(env_name, "bin", "activate")

    # Generate a random UUID
    unique_id = str(uuid.uuid4())
    input_file = os.path.join(current_path, f"event_{unique_id}.txt")
    with open(input_file, "w") as fd:
        fd.write(text)

    output_file = os.path.join(current_path, f"event_output_{unique_id}.joblib")
    subprocess.call(f"source {venv_activation_script} && python {allen_script} --lang {lang} --action event \
        --input {input_file} --output {output_file}", shell=True, executable="/bin/bash")

    #result = pd.read_pickle(output_file)
    result = joblib.load(output_file)

    os.remove(output_file)
    os.remove(input_file)

    return result

def extract_semantic_role_links(lang, text):

    venv_activation_script = os.path.join(env_name, "bin", "activate")

    # Generate a random UUID
    unique_id = str(uuid.uuid4())
    input_file = os.path.join(current_path, f"srlink_{unique_id}.txt")
    with open(input_file, "w") as fd:
        fd.write(text)

    output_file = os.path.join(current_path, f"srlink_output_{unique_id}.joblib")
    subprocess.call(f"source {venv_activation_script} && python {allen_script} --lang {lang} --action srlink \
            --input {input_file} --output {output_file}", shell=True, executable="/bin/bash")

    # result = pd.read_pickle(output_file)
    result = joblib.load(output_file)

    os.remove(output_file)
    os.remove(input_file)

    return result

def extract_objectal_links(lang, text):

    venv_activation_script = os.path.join(env_name, "bin", "activate")

    # Generate a random UUID
    unique_id = str(uuid.uuid4())
    working_dir = os.getcwd()
    input_file = os.path.join(working_dir, f"olink_{unique_id}.txt")
    with open(input_file, "w") as fd:
        fd.write(text)

    output_file = os.path.join(working_dir, f"olink_output_{unique_id}.joblib")
    # print(f"source {venv_activation_script} && python {allen_script} --lang {lang} --action olink \
    #            --input {input_file} --output {output_file}")

    subprocess.call(f"source {venv_activation_script} && python {allen_script} --lang {lang} --action olink \
            --input {input_file} --output {output_file}", shell=True, executable="/bin/bash")

    # result = pd.read_pickle(output_file)
    result = joblib.load(output_file)

    os.remove(output_file)
    os.remove(input_file)

    return result

def extract_srl(lang, text):
    venv_activation_script = os.path.join(env_name, "bin", "activate")

    # Generate a random UUID
    unique_id = str(uuid.uuid4())
    working_dir = os.getcwd()
    input_file = os.path.join(working_dir, f"srl_{unique_id}.txt")
    with open(input_file, "w") as fd:
        fd.write(text)

    output_file = os.path.join(working_dir, f"srl_output_{unique_id}.joblib")
    # print(f"source {venv_activation_script} && python {allen_script} --lang {lang} --action olink \
    #            --input {input_file} --output {output_file}")

    subprocess.call(f"source {venv_activation_script} && python {allen_script} --lang {lang} --action srl \
                --input {input_file} --output {output_file}", shell=True, executable="/bin/bash")

    # result = pd.read_pickle(output_file)
    result = joblib.load(output_file)

    os.remove(output_file)
    os.remove(input_file)

    return result


