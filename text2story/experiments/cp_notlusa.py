import os
import shutil

# List of labels to exclude
exclude_labels = [
   "AGGRESSIVEVOICEDAILY_20041203.1959",
        "MARKETVIEW_20050208.2033",
        "MARKETVIEW_20050127.0716",
        "CNN_IP_20030403.1600.00-1",
        "CNN_ENG_20030416_160804.4",
        "AFP_ENG_20030417.0764",
        "fsh_29586",
        "CNN_ENG_20030605_193002.8",
        "CNN_IP_20030329.1600.00-3",
        "CNN_IP_20030410.1600.03-1",
        "CNN_ENG_20030602_105829.2",
        "CNN_ENG_20030411_193701.3",
        "AGGRESSIVEVOICEDAILY_20041215.2302",
        "fsh_29344",
        "misc.taxes_20050218.1250",
        "APW_ENG_20030324.0768",
        "CNN_ENG_20030627_065846.3",
        "CNN_ENG_20030526_180540.6",
        "CNN_ENG_20030612_072835.2",
        "APW_ENG_20030322.0119",
        "APW_ENG_20030326.0190",
        "MARKETVIEW_20041217.0801",
        "CNN_ENG_20030312_083725.3",
        "CNN_ENG_20030331_193655.14",
        "CNN_IP_20030405.1600.01-2",
        "CNN_ENG_20030507_060023.1",
        "AGGRESSIVEVOICEDAILY_20050107.2012",
        "CNN_ENG_20030417_073039.2",
        "MARKETVIEW_20050126.0711",
        "AGGRESSIVEVOICEDAILY_20050224.2252",
        "OIADVANTAGE_20050203.1000",
        "CNN_ENG_20030619_115954.4",
        "CNN_ENG_20030620_095840.4",
        "CNNHL_ENG_20030513_183907.5",
        "APW_ENG_20030519.0548",
        "CNN_IP_20030422.1600.05",
        "CNNHL_ENG_20030611_133445.24",
        "CNNHL_ENG_20030603_230307.3",
        "CNN_LE_20030504.1200.01",
        "CNN_IP_20030410.1600.03-2",
        "CNNHL_ENG_20030403_133453.21",
        "MARKETVIEW_20050222.1919",
        "APW_ENG_20030422.0469",
        "CNN_ENG_20030509_123601.13",
        "CNNHL_ENG_20030304_142751.10",
        "CNN_IP_20030329.1600.00-5",
        "CNN_ENG_20030428_173654.13",
        "CNN_ENG_20030612_160005.13",
        "rec.travel.usa-canada_20050128.0121",
        "MARKETVIEW_20050215.1858",
        "BACONSREBELLION_20050227.1238",
        "CNN_ENG_20030622_173306.9",
        "AFP_ENG_20030630.0271",
        "CNNHL_ENG_20030610_230438.14",
        "CNN_CF_20030303.1900.00",
        "APW_ENG_20030406.0191",
        "APW_ENG_20030311.0775",
        "CNN_ENG_20030423_180539.2",
        "CNN_ENG_20030618_150128.5",
        "MARKETVIEW_20050212.1717",
        "CNN_ENG_20030312_223733.14",
        "CNN_ENG_20030610_133041.17",
        "CNN_ENG_20030528_165958.16",
        "CNN_ENG_20030428_130651.4",
        "CNN_IP_20030330.1600.06",
        "AFP_ENG_20030319.0879",
        "CNN_ENG_20030621_115841.16",
        "MARKETVIEW_20050208.2059",
        "fsh_29138",
        "CNN_ENG_20030418_063040.1",
        "marcellapr_20050211.2013",
        "MARKBACKER_20041119.1002",
        "CNN_ENG_20030418_163834.14",
        "CNN_ENG_20030617_173115.14",
        "soc.history.war.world-war-ii_20050127.2403",
        "CNN_ENG_20030408_153616.9",
        "CNN_IP_20030405.1600.01-1",
        "CNN_ENG_20030529_085826.10",
        "CNN_ENG_20030605_085831.13",
        "CNN_IP_20030412.1600.05"
]

source_dir = "ace/"  # Source directory
output_dir = "ace_test"  # Destination directory

# Create the destination directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List files in the source directory
files = os.listdir(source_dir)

# Iterate through the files and copy them to the destination directory if they don't start with excluded labels
for file_name in files:
    if not any(file_name.startswith(label) for label in exclude_labels):
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(output_dir, file_name)
        shutil.copy2(source_path, destination_path)

print("Files copied successfully.")

