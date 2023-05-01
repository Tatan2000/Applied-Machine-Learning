import os, yaml, time
import subprocess

from src.app import app
from src.score import score

##########################################################################################################################


config_file_path = os.path.join('config', 'test_config', 'integration.yaml')

with open(config_file_path) as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

app_path = config['app_path']

input_text = config['input_text']

def test_app():
    # start the flask app using command_line
    cmd = 'python ' + app_path + ' &'
    os.system(cmd)

    time.sleep(5)

    
    endpoint = '/score'
    response_get = app.test_client().get(endpoint)
   
    assert (response_get.status_code == 405)

    
    
    sms_text = input_text
    response_post = app.test_client().post(endpoint, data={"sms_text":sms_text})

    assert (response_post.status_code == 200)
    
    output_json_str = response_post.data.decode()
    assert "prediction" in output_json_str
    assert "propensity" in output_json_str
    
    assert (type(output_json_str) == str)
    assert ("\"prediction\":true" in output_json_str)


    # stop the Flask app using command line
    os.system('kill $(lsof -t -i:5000)')

##########################################################################################################################
