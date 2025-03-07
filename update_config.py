import os
import toml

# Load the existing TOML file
config_file_path = 'path/to/your/config.toml'
with open(config_file_path, 'r') as file:
    config = toml.load(file)

# Update the PORT value with the environment variable
port = os.getenv('PORT')
if port:
    config['server']['port'] = int(port)

# Write the updated config back to the TOML file
with open(config_file_path, 'w') as file:
    toml.dump(config, file)
